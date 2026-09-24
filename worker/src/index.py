from typing import Any
import asyncio
import base64
import json
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from workers import WorkerEntrypoint, Response, asgi
from js import Object, fetch
from pyodide.ffi import to_js


# ============================================================
# APP
# ============================================================

APP_VERSION = "1.3.0"
PRIMARY_MODEL_DEFAULT = "gemini-3.8-flash"
FALLBACK_MODEL_DEFAULT = "gemini-3.7-flash"
LAST_RESORT_MODEL_DEFAULT = "gemini-3.5-flash-lite"
MAX_PDF_SIZE = 15 * 1024 * 1024
MAX_EQUIPMENT = 30

app = FastAPI(
    title="PDF Equipment Extractor",
    version=APP_VERSION,
    description="Engineering P&ID equipment extraction using Gemini vision + verification.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PARAMETER_KEYS = [
    "tag_no",
    "equipment_name",
    "type",
    "diameter",
    "diameter_od_id",
    "length",
    "length_remarks",
    "height",
    "insulation",
    "insulation_size",
    "insulation_type",
    "shell_pressure",
    "shell_min_temp",
    "shell_max_temp",
    "tube_pressure",
    "tube_min_temp",
    "tube_max_temp",
    "service_type",
    "service_description",
    "remarks",
]


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    filename: str
    mime_type: str
    pdf_base64: str


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_value(value: Any) -> str:
    if value is None:
        return "-"

    text = str(value).strip()

    if not text:
        return "-"

    if text.lower() in {
        "null",
        "none",
        "n/a",
        "na",
        "not available",
        "not found",
        "unknown",
        "undefined",
        "not applicable",
    }:
        return "-"

    return text


def normalize_equipment(item: dict) -> dict:
    return {key: normalize_value(item.get(key)) for key in PARAMETER_KEYS}


def normalize_equipment_list(items: Any) -> list[dict]:
    if not isinstance(items, list):
        return []

    output = []
    seen_tags = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        normalized = normalize_equipment(item)
        tag = normalized["tag_no"]

        if tag != "-":
            tag_key = tag.upper().replace(" ", "")
            if tag_key in seen_tags:
                continue
            seen_tags.add(tag_key)

        output.append(normalized)

        if len(output) >= MAX_EQUIPMENT:
            break

    return output


# ============================================================
# ENV HELPERS
# ============================================================

def get_env(request: Request) -> Any:
    env = request.scope.get("env")
    if env is None:
        raise HTTPException(
            status_code=500,
            detail="Cloudflare environment is unavailable.",
        )
    return env


def get_binding(env: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(env, name)
    except Exception:
        return default

    if value is None:
        return default

    return value


# ============================================================
# GEMINI SCHEMAS
# ============================================================

EQUIPMENT_PROPERTIES = {
    "tag_no": {"type": "STRING"},
    "equipment_name": {"type": "STRING"},
    "type": {"type": "STRING"},
    "diameter": {"type": "STRING"},
    "diameter_od_id": {"type": "STRING"},
    "length": {"type": "STRING"},
    "length_remarks": {"type": "STRING"},
    "height": {"type": "STRING"},
    "insulation": {"type": "STRING"},
    "insulation_size": {"type": "STRING"},
    "insulation_type": {"type": "STRING"},
    "shell_pressure": {"type": "STRING"},
    "shell_min_temp": {"type": "STRING"},
    "shell_max_temp": {"type": "STRING"},
    "tube_pressure": {"type": "STRING"},
    "tube_min_temp": {"type": "STRING"},
    "tube_max_temp": {"type": "STRING"},
    "service_type": {"type": "STRING"},
    "service_description": {"type": "STRING"},
    "remarks": {"type": "STRING"},
}

EQUIPMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "equipment": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": EQUIPMENT_PROPERTIES,
                "required": PARAMETER_KEYS,
            },
        }
    },
    "required": ["equipment"],
}

DISCOVERY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "equipment": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "tag_no": {"type": "STRING"},
                    "equipment_name": {"type": "STRING"},
                    "type": {"type": "STRING"},
                },
                "required": ["tag_no", "equipment_name", "type"],
            },
        }
    },
    "required": ["equipment"],
}


# ============================================================
# PROMPTS
# ============================================================

DISCOVERY_PROMPT = r"""
You are the EQUIPMENT DISCOVERY stage of a professional engineering P&ID
extraction system.

Inspect EVERY PAGE of the supplied PDF visually. Your first priority is to
find ALL physical process equipment shown on the drawing. Do not extract
parameters yet.

A process equipment item normally has one or more of these clues:
- an equipment tag such as E-2, 1E-2, C-3, 1C-3, V-101, P-101, etc.;
- an equipment name/title such as COOLER, SEPARATOR, DRUM, VESSEL, COLUMN,
  TANK, PUMP, COMPRESSOR, HEATER, REACTOR;
- a recognizable equipment symbol/body with a tag or title nearby;
- a dedicated equipment data block containing design information.

IMPORTANT: equipment tags vary by project. Do NOT assume that only one tag
prefix is valid. Look at the actual drawing conventions.

INCLUDE physical major/process equipment such as:
- heat exchangers and coolers
- separators and drums
- vessels and columns
- tanks
- pumps
- compressors
- heaters/furnaces
- reactors
- other clearly represented major process equipment

EXCLUDE:
- valves
- instruments
- piping and line numbers
- fittings and flanges
- individual nozzles
- pipe supports
- dimensions/elevations by themselves
- utility or line labels that are not equipment

CRITICAL:
1. Search the whole page, not only the center of the drawing.
2. Equipment can be drawn as a symbol/body with its tag above, below, or
   beside it.
3. A tag/name close to piping can still identify equipment if the tag/title
   is clearly associated with the equipment body or equipment data block.
4. Do not require the equipment name to be present. If a clear equipment tag
   and equipment body exist, return it with equipment_name="-".
5. Do not require dimensions, pressure or temperature to identify equipment.
6. Do not return piping, valves or instruments merely because they have tags.
7. Do not invent tags. Copy visible tag text exactly.

For each equipment candidate return:
- tag_no: exact visible equipment tag; if no tag is visible but the equipment
  is unmistakably identified by an explicit title, use the title only.
- equipment_name: exact visible equipment title/name, otherwise "-".
- type: simple type such as COOLER, HEAT EXCHANGER, SEPARATOR, DRUM, VESSEL,
  COLUMN, TANK, PUMP, COMPRESSOR, HEATER, REACTOR, or "-".

Do NOT extract pressure, temperature, diameter, length, height, duty or
service in this stage.

Before returning, scan the entire PDF once more for missed major equipment.
Return JSON ONLY.
"""

# A recovery prompt is intentionally broader and is only called when the
# primary discovery pass returns zero equipment. This prevents a strict
# first-pass interpretation from causing a false empty result.
RECOVERY_DISCOVERY_PROMPT = r"""
You are a SECOND-PASS VISUAL EQUIPMENT FINDER for an engineering P&ID.

The previous detector found zero equipment. Do NOT accept that conclusion.
Reinspect the entire PDF visually from the beginning.

Find every physical major process equipment body/symbol on the sheet, even if
its equipment title is separate from the symbol or partly surrounded by
piping. Search for combinations of:
- equipment-shaped bodies (vessels, exchangers, drums, columns, tanks);
- equipment tags;
- equipment titles;
- dedicated design-data blocks;
- explicit equipment dimensions attached to a body.

Typical project tag examples include E-2, 1E-2, C-3, 1C-3, V-101, P-101,
but these are examples only; use the actual drawing.

Return major physical equipment only. Do NOT return valves, instruments,
pipe lines, line numbers, fittings, flanges, nozzles, supports, elevations,
or isolated dimensions.

If a physical equipment body is clearly present and its tag is readable,
return it even when the title is missing. Use equipment_name="-" when needed.
Never invent a tag or name.

Return JSON ONLY using the requested schema.
"""

EXTRACTION_PROMPT_TEMPLATE = r"""
You are the FINAL engineering P&ID extraction and verification engine.

The PDF is the ONLY source of truth.
You have been given a preliminary equipment list detected from this same PDF.
Use it as a controlled checklist, but independently inspect the PDF again
before assigning ANY value.

PRELIMINARY EQUIPMENT LIST:
{equipment_list}

Your job is to return exactly the real equipment records from this list,
with the requested parameters.

============================================================
CORE RULE — VISUAL ASSOCIATION
============================================================

A value belongs to an equipment item ONLY when the drawing explicitly and
visually associates that value with that exact equipment.

Never take a value merely because it is nearby, connected by a pipe, printed
inside another equipment block, or visually prominent.

When association is uncertain, return "-".

============================================================
DIMENSIONS
============================================================

DIAMETER:
Use only the equipment's own explicitly labelled diameter.
Examples: 10'-0" Ø, 10'-0" DIA, 10'-0" DIAMETER.

NEVER use a pipe/nozzle size such as 2", 3", 4", 6", 8", 10", 12" as the
equipment diameter unless the drawing explicitly labels it as the equipment
diameter.

LENGTH:
Use only the equipment/vessel length explicitly associated with that item.
For 10'-0" Ø × 9'-0" T-T:
- diameter = 10'-0"
- length = 9'-0"
- length_remarks = T-T

For 10'-0" Ø × 40'-6" T/T:
- diameter = 10'-0"
- length = 40'-6"
- length_remarks = T/T

HEIGHT:
Only populate when an explicit equipment height is shown.
Never substitute an elevation, nozzle projection or piping dimension.

============================================================
PRESSURE / TEMPERATURE
============================================================

Use ONLY design pressure and design temperature belonging to the exact
identified equipment.

For vessels, separators and drums:
- vessel design pressure -> shell_pressure
- vessel design temperature -> shell_min_temp or shell_max_temp only when
the drawing explicitly distinguishes minimum/maximum.
- otherwise put the stated design temperature into shell_max_temp only if
that field is clearly the applicable design maximum. If the drawing does not
support the distinction, preserve the value in remarks and use "-" for the
ambiguous temperature field.
- tube fields = "-"

For coolers / heat exchangers:
KEEP SHELL AND TUBE DATA SEPARATE.
Do not swap shell and tube values.

============================================================
INSULATION
============================================================

If the exact equipment data explicitly says NONE -> insulation = "No".
If explicitly insulated -> "Yes".
Otherwise -> "-".

============================================================
SERVICE
============================================================

Only use values explicitly supported by the drawing.
Do not infer service/fluid from the equipment name.

============================================================
DUTY / OTHER DATA
============================================================

If DUTY is explicitly shown and there is no dedicated template field,
keep it in remarks.

============================================================
ANTI-HALLUCINATION RULES
============================================================

NEVER:
- estimate
- calculate
- convert units unless the drawing itself gives the converted value
- copy a value from another equipment
- use a pipe/nozzle dimension as equipment dimension
- use a piping pressure/rating as equipment design pressure
- invent missing min/max temperatures

Use exactly "-" for missing, unreadable, ambiguous or unsupported values.

============================================================
FINAL TWO-PASS CHECK
============================================================

For EACH equipment record, silently check:
1. tag is exactly from the drawing;
2. name belongs to that tag;
3. diameter belongs to the equipment itself;
4. length belongs to the equipment itself;
5. pressure belongs to that equipment;
6. temperature belongs to that equipment;
7. shell/tube values were not mixed;
8. no value came from nearby piping/nozzles;
9. uncertain values are "-".

Return JSON ONLY according to the supplied schema.
"""



# ============================================================
# DIRECT EXTRACTION RECOVERY PROMPT
# ============================================================

DIRECT_EXTRACTION_PROMPT = r"""
You are the PRIMARY VISUAL ENGINEERING P&ID EXTRACTION ENGINE.

IMPORTANT: Do NOT perform a separate discovery step. Inspect the supplied
PDF yourself and directly identify the real major process equipment AND its
parameters in one visual pass.

The PDF page is the only source of truth.

============================================================
EQUIPMENT SCOPE
============================================================

Find physical major/process equipment that is actually shown on this sheet:
- heat exchangers / coolers
- separators / drums / vessels
- columns
- tanks
- pumps / compressors
- heaters / furnaces / reactors
- other clearly identifiable major process equipment

DO NOT count:
- valves
- instruments
- piping or line numbers
- fittings / flanges
- individual nozzles
- pipe supports
- elevations
- isolated dimensions
- equipment merely referenced by text when its physical body/data is not
  actually present on this sheet

A valid equipment record normally has a visible equipment body/symbol and/or
an equipment data block, with a tag or explicit equipment title associated
with it. The tag may be above, below, beside, or inside the equipment area.

Do not invent tags. Copy visible tag text exactly.

============================================================
VISUAL ASSOCIATION RULE
============================================================

Every parameter MUST belong to the same equipment as the tag.

NEVER use:
- pipe size as equipment diameter
- nozzle size as equipment diameter
- valve rating as equipment pressure
- piping pressure as equipment design pressure
- elevation as equipment height
- dimensions belonging to another equipment
- values from a neighboring equipment data block

If you cannot visually prove the association, return "-".

============================================================
DIMENSIONS
============================================================

For vessels/drums/separators/exchangers, use only dimensions explicitly
associated with that equipment.

Example:
10'-0" Ø × 9'-0" T-T
=> diameter = 10'-0"
=> length = 9'-0"
=> length_remarks = T-T

Example:
10'-0" Ø × 40'-6" T/T
=> diameter = 10'-0"
=> length = 40'-6"
=> length_remarks = T/T

Never use a nearby pipe/nozzle dimension as diameter or length.

============================================================
PRESSURE / TEMPERATURE
============================================================

Use only explicitly labelled equipment design values.

For vessels, drums and separators:
- design pressure -> shell_pressure
- design temperature -> shell_max_temp ONLY when it is explicitly a design
  temperature and there is no min/max ambiguity
- tube fields -> "-"

For exchangers/coolers:
- keep shell and tube sides separate
- do not swap shell and tube values

If a value is ambiguous, use "-".

============================================================
OTHER DATA
============================================================

Insulation: use only explicit information such as NONE.
Service: use only explicit information; do not infer fluid from equipment name.
Duty: if explicitly shown but there is no dedicated field, put it in remarks.

============================================================
ANTI-HALLUCINATION
============================================================

Never calculate, estimate, infer, convert, or borrow values.
Missing/unreadable/ambiguous values MUST be "-".

Before returning, perform a visual cross-check for every record:
1. tag belongs to the physical equipment;
2. name belongs to that tag;
3. dimensions belong to that equipment;
4. pressure belongs to that equipment;
5. temperature belongs to that equipment;
6. shell/tube are not mixed;
7. no nearby piping/nozzle values were copied;
8. only equipment physically represented on this sheet is included.

Return JSON ONLY using the supplied schema.
"""

# ============================================================
# GEMINI CALL HELPERS
# ============================================================

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
RETRY_DELAYS = [2.0, 5.0]


def model_is_gemini_3(model: str) -> bool:
    return model.startswith("gemini-3.")


def model_config(model: str, high_quality: bool = True) -> dict:
    cfg = {
        "responseMimeType": "application/json",
    }
    # Thinking-level is supported by Gemini 3.x. The older 3.5 Flash-Lite
    # fallback must not receive unsupported Gemini 3 thinking parameters.
    if model_is_gemini_3(model):
        cfg["thinkingConfig"] = {"thinkingLevel": "high" if high_quality else "low"}
        cfg["media_resolution"] = "MEDIA_RESOLUTION_HIGH" if high_quality else "MEDIA_RESOLUTION_MEDIUM"
    else:
        cfg["media_resolution"] = "MEDIA_RESOLUTION_HIGH" if high_quality else "MEDIA_RESOLUTION_MEDIUM"
    return cfg


async def sleep_retry(seconds: float) -> None:
    await asyncio.sleep(seconds + random.uniform(0.0, 0.75))


def extract_error_message(response_text: str) -> str:
    try:
        data = json.loads(response_text)
        return str(
            data.get("error", {}).get("message", response_text)
        )
    except Exception:
        return response_text or "Unknown Gemini API error."


async def gemini_request(
    api_key: str,
    model: str,
    pdf_base64: str,
    prompt: str,
    schema: dict,
) -> tuple[dict, int]:
    endpoint = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_base64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            **model_config(model, high_quality=True),
            "responseSchema": schema,
        },
    }

    fetch_options = to_js(
        {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(request_body),
        },
        dict_converter=Object.fromEntries,
    )

    last_status = 502
    last_message = "Gemini request failed."

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = await fetch(endpoint, fetch_options)
            status_code = int(response.status)
            response_text = str(await response.text())
        except Exception as exc:
            status_code = 502
            response_text = str(exc)

        if 200 <= status_code < 300:
            try:
                data = json.loads(response_text)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini returned invalid JSON: {exc}",
                )
            return data, status_code

        last_status = status_code
        last_message = extract_error_message(response_text)

        if status_code not in RETRYABLE_STATUS or attempt >= len(RETRY_DELAYS):
            break

        await sleep_retry(RETRY_DELAYS[attempt])

    raise HTTPException(
        status_code=503 if last_status in RETRYABLE_STATUS else 502,
        detail=(
            f"Gemini API error ({last_status}): {last_message} "
            "The request was retried automatically."
        ),
        headers={"X-Gemini-Model": model},
    )


async def gemini_with_fallbacks(
    api_key: str,
    models: list[str],
    pdf_base64: str,
    prompt: str,
    schema: dict,
) -> tuple[dict, str]:
    """Try models in order. Each model gets bounded retry on transient errors.

    Primary remains Gemini 3.8 Flash. The fallbacks exist for temporary 503
    capacity spikes so a free-tier capacity event does not break the workflow.
    """
    last_error = None
    for index, model in enumerate(models):
        if not model:
            continue
        try:
            data, _ = await gemini_request(
                api_key, model, pdf_base64, prompt, schema
            )
            return data, model
        except HTTPException as exc:
            last_error = exc
            # Only capacity/server errors should move to another model.
            if exc.status_code not in {408, 429, 500, 502, 503, 504}:
                raise
            # Continue to the next model.
            continue

    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=503, detail="No Gemini model is configured.")


def parse_generated_json(gemini_data: dict) -> dict:
    candidates = gemini_data.get("candidates") or []
    if not candidates:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no candidates.",
        )

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    generated_text = ""
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            generated_text = part["text"]
            break

    if not generated_text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no structured text.",
        )

    try:
        return json.loads(generated_text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid structured JSON: {exc}",
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "PDF Equipment Extractor",
        "status": "ok",
        "version": APP_VERSION,
        "pipeline": "discovery + recovery + direct-extraction + verification",
    }


@app.get("/health")
async def health(request: Request):
    env = get_env(request)
    model = str(get_binding(env, "GEMINI_MODEL", PRIMARY_MODEL_DEFAULT)).strip()
    fallback = str(
        get_binding(env, "GEMINI_FALLBACK_MODEL", FALLBACK_MODEL_DEFAULT)
    ).strip()
    last_resort = str(
        get_binding(env, "GEMINI_LAST_RESORT_MODEL", LAST_RESORT_MODEL_DEFAULT)
    ).strip()

    return {
        "status": "ok",
        "model": model or PRIMARY_MODEL_DEFAULT,
        "fallback_model": fallback or FALLBACK_MODEL_DEFAULT,
        "last_resort_model": last_resort or LAST_RESORT_MODEL_DEFAULT,
        "pdf_direct_vision": True,
        "media_resolution": "high",
        "thinking_level": "high",
        "pipeline": "discovery + recovery + direct-extraction + verification",
        "extractor_version": APP_VERSION,
    }


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request):
    env = get_env(request)

    api_key = get_binding(env, "GEMINI_API_KEY")
    primary_model = str(
        get_binding(env, "GEMINI_MODEL", PRIMARY_MODEL_DEFAULT)
    ).strip() or PRIMARY_MODEL_DEFAULT
    fallback_model = str(
        get_binding(env, "GEMINI_FALLBACK_MODEL", FALLBACK_MODEL_DEFAULT)
    ).strip() or FALLBACK_MODEL_DEFAULT
    last_resort_model = str(
        get_binding(env, "GEMINI_LAST_RESORT_MODEL", LAST_RESORT_MODEL_DEFAULT)
    ).strip() or LAST_RESORT_MODEL_DEFAULT
    model_chain = []
    for candidate in [primary_model, fallback_model, last_resort_model]:
        if candidate and candidate not in model_chain:
            model_chain.append(candidate)

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    if payload.mime_type.lower().strip() != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only application/pdf is supported.",
        )

    try:
        pdf_bytes = base64.b64decode(payload.pdf_base64, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF Base64 data.",
        )

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF file is empty.")

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF is larger than the 15 MB limit.",
        )

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    # --------------------------------------------------------
    # PASS 1: EQUIPMENT DISCOVERY
    # --------------------------------------------------------
    discovery_data, discovery_model = await gemini_with_fallbacks(
        api_key,
        model_chain,
        payload.pdf_base64,
        DISCOVERY_PROMPT,
        DISCOVERY_SCHEMA,
    )

    discovered = discovery_data.get("equipment", [])
    if not isinstance(discovered, list):
        raise HTTPException(
            status_code=502,
            detail="Gemini discovery result is invalid.",
        )

    discovered_clean = []
    seen = set()

    for item in discovered:
        if not isinstance(item, dict):
            continue

        tag = normalize_value(item.get("tag_no"))
        name = normalize_value(item.get("equipment_name"))
        eq_type = normalize_value(item.get("type"))

        if tag == "-":
            continue

        key = tag.upper().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)

        discovered_clean.append({
            "tag_no": tag,
            "equipment_name": name,
            "type": eq_type,
        })

        if len(discovered_clean) >= MAX_EQUIPMENT:
            break

    # --------------------------------------------------------
    # DISCOVERY RECOVERY PASS
    # --------------------------------------------------------
    # If the strict first pass returns nothing, run one broader visual pass.
    # This is the key protection against a false "0 equipment" result.
    if not discovered_clean:
        try:
            recovery_data, recovery_model = await gemini_with_fallbacks(
                api_key,
                model_chain,
                payload.pdf_base64,
                RECOVERY_DISCOVERY_PROMPT,
                DISCOVERY_SCHEMA,
            )
            recovery_items = recovery_data.get("equipment", [])
            if isinstance(recovery_items, list):
                seen = set()
                for item in recovery_items:
                    if not isinstance(item, dict):
                        continue

                    tag = normalize_value(item.get("tag_no"))
                    name = normalize_value(item.get("equipment_name"))
                    eq_type = normalize_value(item.get("type"))

                    if tag == "-":
                        continue

                    key = tag.upper().replace(" ", "")
                    if key in seen:
                        continue
                    seen.add(key)

                    discovered_clean.append({
                        "tag_no": tag,
                        "equipment_name": name,
                        "type": eq_type,
                    })

                    if len(discovered_clean) >= MAX_EQUIPMENT:
                        break
        except HTTPException as recovery_error:
            # Keep the original discovery result if the recovery call fails.
            # If both passes fail, the user gets a useful empty result rather
            # than a fabricated equipment record.
            if recovery_error.status_code not in {502, 503}:
                raise

    # --------------------------------------------------------
    # DIRECT EXTRACTION RECOVERY
    # --------------------------------------------------------
    # A discovery-only architecture can fail even when the vision model can
    # actually read the PDF. If both inventory passes return zero, do NOT stop.
    # Ask the model to perform direct full extraction from the PDF.
    direct_recovery_used = False
    direct_recovery_model = None
    direct_equipment = []

    if not discovered_clean:
        direct_recovery_used = True
        direct_data, direct_recovery_model = await gemini_with_fallbacks(
            api_key,
            model_chain,
            payload.pdf_base64,
            DIRECT_EXTRACTION_PROMPT,
            EQUIPMENT_SCHEMA,
        )
        direct_equipment = normalize_equipment_list(
            direct_data.get("equipment", [])
        )

        # Use directly extracted records as the controlled checklist for the
        # verification pass. This avoids returning a false zero.
        for item in direct_equipment:
            if item.get("tag_no") == "-":
                continue
            key = item["tag_no"].upper().replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            discovered_clean.append({
                "tag_no": item["tag_no"],
                "equipment_name": item["equipment_name"],
                "type": item["type"],
            })

    # If every recovery attempt genuinely found nothing, return a diagnostic
    # result. This is now a last resort rather than the normal path.
    if not discovered_clean:
        return {
            "status": "ok",
            "filename": payload.filename,
            "equipment_count": 0,
            "equipment": [],
            "pipeline": "discovery + recovery + direct-extraction + verification",
            "model_used": direct_recovery_model or discovery_model,
            "discovery_model": discovery_model,
            "direct_recovery_used": direct_recovery_used,
            "verification": "not_run",
            "diagnostic": "No physical major equipment was identified after discovery, recovery, and direct visual extraction passes.",
        }

    # --------------------------------------------------------
    # PASS 2: EXTRACTION + INDEPENDENT VERIFICATION
    # --------------------------------------------------------
    extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        equipment_list=json.dumps(discovered_clean, ensure_ascii=False)
    )

    extraction_data, extraction_model = await gemini_with_fallbacks(
        api_key,
        model_chain,
        payload.pdf_base64,
        extraction_prompt,
        EQUIPMENT_SCHEMA,
    )

    raw_equipment = extraction_data.get("equipment", [])
    equipment = normalize_equipment_list(raw_equipment)

    # Controlled ordering: follow the discovery order instead of allowing the
    # model to reorder records unpredictably.
    by_tag = {
        item["tag_no"].upper().replace(" ", ""): item
        for item in equipment
        if item.get("tag_no") != "-"
    }

    ordered = []
    for discovered_item in discovered_clean:
        key = discovered_item["tag_no"].upper().replace(" ", "")
        item = by_tag.get(key)
        if item is not None:
            # Preserve discovery identity if the extraction pass altered it.
            item["tag_no"] = discovered_item["tag_no"]
            if item["equipment_name"] == "-":
                item["equipment_name"] = discovered_item["equipment_name"]
            if item["type"] == "-":
                item["type"] = discovered_item["type"]
            ordered.append(item)

    # If extraction omitted a discovered item, create a blank record rather
    # than silently pretending the item did not exist.
    existing_keys = {
        item["tag_no"].upper().replace(" ", "")
        for item in ordered
    }

    for discovered_item in discovered_clean:
        key = discovered_item["tag_no"].upper().replace(" ", "")
        if key not in existing_keys:
            blank = {key: "-" for key in PARAMETER_KEYS}
            blank["tag_no"] = discovered_item["tag_no"]
            blank["equipment_name"] = discovered_item["equipment_name"]
            blank["type"] = discovered_item["type"]
            ordered.append(blank)

    return {
        "status": "ok",
        "filename": payload.filename,
        "equipment_count": len(ordered),
        "equipment": ordered,
        "pipeline": "discovery + recovery + direct-extraction + verification",
        "model_used": extraction_model,
        "discovery_model": discovery_model,
        "direct_recovery_used": direct_recovery_used,
        "direct_recovery_model": direct_recovery_model,
        "verification": "completed",
    }


# ============================================================
# CLOUDFLARE PYTHON WORKER ENTRYPOINT / CORS
# ============================================================

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        origin = request.headers.get("Origin")

        allowed_origins = {
            "https://hendriseptian.github.io",
            "http://localhost:8787",
            "http://127.0.0.1:8787",
        }

        allow_origin = (
            origin
            if origin in allowed_origins
            else "https://hendriseptian.github.io"
        )

        cors_headers = {
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }

        if request.method == "OPTIONS":
            requested_headers = request.headers.get(
                "Access-Control-Request-Headers"
            )
            if requested_headers:
                cors_headers["Access-Control-Allow-Headers"] = requested_headers

            requested_method = request.headers.get(
                "Access-Control-Request-Method"
            )
            if requested_method:
                cors_headers["Access-Control-Allow-Methods"] = (
                    requested_method + ", OPTIONS"
                )

            return Response("", status=204, headers=cors_headers)

        try:
            response = await asgi.fetch(app, request, self.env)

            response.headers.set(
                "Access-Control-Allow-Origin",
                allow_origin,
            )
            response.headers.set(
                "Access-Control-Allow-Methods",
                "GET, POST, OPTIONS",
            )
            response.headers.set(
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            )
            response.headers.set(
                "Access-Control-Max-Age",
                "86400",
            )
            response.headers.set("Vary", "Origin")

            return response

        except Exception as exc:
            error_body = json.dumps(
                {
                    "detail": "Worker internal error.",
                    "error": str(exc),
                }
            )

            return Response(
                error_body,
                status=500,
                headers={
                    **cors_headers,
                    "Content-Type": "application/json",
                },
            )
