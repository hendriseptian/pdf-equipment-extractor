from typing import Any
import base64
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from workers import asgi
from js import Object, fetch
from pyodide.ffi import to_js


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PDF Equipment Extractor",
    version="0.2.0",
    description="Extract engineering equipment data from PDF drawings using Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PDF_SIZE = 15 * 1024 * 1024

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
# HELPERS
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
        "not specified",
        "not applicable",
    }:
        return "-"

    return text


def normalize_equipment(item: dict) -> dict:
    result = {
        key: normalize_value(item.get(key))
        for key in PARAMETER_KEYS
    }

    # Never allow an accidental null-like value into the final response.
    for key, value in result.items():
        result[key] = normalize_value(value)

    return result


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
# GEMINI STRUCTURED OUTPUT SCHEMA
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

GEMINI_SCHEMA = {
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


# ============================================================
# EXTRACTION PROMPT V2
# ============================================================

EXTRACTION_PROMPT = r"""
You are a highly precise engineering drawing extraction system.

Your job is NOT to summarize the drawing. Your job is to extract equipment
records and only the engineering data that is explicitly associated with
those equipment records.

============================================================
A. FIRST: BUILD AN EQUIPMENT INVENTORY
============================================================

Before filling any fields, inspect the ENTIRE PDF and identify every major
piece of process/mechanical equipment that has a clear equipment tag or
clear equipment identity.

Examples of equipment:
- vessel
- separator
- drum
- tank
- heat exchanger
- cooler
- heater
- pump
- compressor
- column
- reactor

Do NOT count these as equipment:
- valves
- instruments
- piping
- pipe lines
- fittings
- flanges
- reducers
- elbows
- control valves
- small inline components

Each equipment must appear exactly once in the output.

============================================================
B. MOST IMPORTANT: ASSOCIATE DATA WITH THE CORRECT EQUIPMENT
============================================================

For each equipment, identify the LOCAL text block, callout, equipment data
block, specification block, or label that belongs to that equipment.

A dedicated equipment data/specification block associated with an equipment
tag is authoritative for that equipment.

If the drawing contains an equipment summary/table/block near the title or
upper part of the sheet, use the values in that block when they are clearly
associated with the equipment tag.

DO NOT take a number merely because it is visually close on the page.

DO NOT copy a value from another equipment.

DO NOT use values from process piping, instrument bubbles, valve tags, or
nearby equipment unless the drawing explicitly associates the value with the
target equipment.

============================================================
C. CRITICAL DIMENSION RULE
============================================================

Equipment diameter is NOT the same thing as nearby pipe/line size.

Examples:
- A pipe labeled 2", 3", 4", 6", etc. is NOT an equipment diameter.
- A vessel label such as 10'-0" Ø is an equipment diameter.
- A vessel label such as 10'-0" Ø x 40'-6" T/T means:
    diameter = 10'-0"
    length = 40'-6"
    length_remarks = T/T

Only put a diameter in the diameter field when the drawing explicitly gives
an equipment/vessel diameter.

Never infer equipment diameter from connected pipe sizes.

============================================================
D. CRITICAL TAG RULE
============================================================

Read the equipment tag from the equipment's own label/data block.

Do NOT substitute another nearby tag.

If the drawing has similar-looking tags, verify the tag against the
associated equipment name and local data block before returning it.

============================================================
E. PRESSURE AND TEMPERATURE RULE
============================================================

Keep SHELL SIDE and TUBE SIDE completely separate.

For a heat exchanger/cooler:
- shell_pressure = explicitly stated shell design pressure
- shell_min_temp = explicitly stated shell minimum temperature, if stated
- shell_max_temp = explicitly stated shell maximum/design temperature, if
  the drawing clearly identifies it as maximum/design temperature
- tube_pressure = explicitly stated tube design pressure
- tube_min_temp = explicitly stated tube minimum temperature, if stated
- tube_max_temp = explicitly stated tube maximum/design temperature, if
  the drawing clearly identifies it as maximum/design temperature

If the drawing only says "DESIGN TEMP" with one value and does not identify
minimum or maximum, DO NOT invent a minimum/maximum classification.
Put the exact design-temperature statement in remarks and use "-" for both
min/max fields unless the drawing explicitly identifies which one it is.

For vessels, drums, separators, tanks, columns, etc.:
- use the explicitly stated vessel design pressure in shell_pressure
- use temperature fields only when their min/max meaning is explicitly
  supported
- tube-side fields are "-" unless the drawing explicitly provides a tube
  side

Never mix shell and tube values.

============================================================
F. INSULATION RULE
============================================================

If the equipment data explicitly says:
- NONE -> insulation = "No"
- insulation is provided -> insulation = "Yes"

Only extract insulation size/type when explicitly stated.

============================================================
G. DUTY RULE
============================================================

If DUTY is explicitly shown but there is no dedicated DUTY field, preserve
it in remarks.

Do not convert or calculate duty.

============================================================
H. EXACT VALUE RULE
============================================================

Preserve the engineering value and units as shown whenever possible.

Do not calculate conversions.
Do not normalize feet/inches into meters.
Do not change PSIG into kg/cm2 unless both are explicitly shown.
Do not calculate missing dimensions.

============================================================
I. MISSING DATA RULE
============================================================

If a field is not explicitly available for that equipment, return "-".

Never use:
- null
- none
- unknown
- N/A
- not found

Use "-" instead.

============================================================
J. SERVICE RULE
============================================================

service_type and service_description must come only from explicit drawing
information.

Do not guess a service or fluid from an equipment name.

============================================================
K. SELF-CHECK BEFORE RETURNING JSON
============================================================

Before returning the JSON, silently perform these checks:

1. Did I identify every major equipment?
2. Does every equipment tag belong to the equipment name beside it?
3. Did I accidentally use a pipe size as an equipment diameter?
4. Did I accidentally use a nearby equipment's dimension?
5. Did I mix shell-side and tube-side pressure/temperature?
6. Did I invent any missing value?
7. Are all unavailable fields exactly "-"?
8. If a dedicated equipment data block exists, did I prioritize it over
   unrelated graphical values elsewhere on the P&ID?

If any answer is uncertain, return "-" rather than guessing.

Return JSON only according to the supplied schema.
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "PDF Equipment Extractor",
        "status": "ok",
        "version": "0.2.0",
    }


@app.get("/health")
async def health(request: Request):
    env = get_env(request)
    model = get_binding(env, "GEMINI_MODEL", "gemini-3.5-flash-lite")

    return {
        "status": "ok",
        "model": str(model),
        "pdf_direct_vision": True,
        "extractor_version": "0.2.0",
    }


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request):
    env = get_env(request)

    api_key = get_binding(env, "GEMINI_API_KEY")
    model = get_binding(env, "GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    model = str(model).strip() or "gemini-3.5-flash-lite"

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if payload.mime_type.lower().strip() != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only application/pdf is supported.",
        )

    try:
        pdf_bytes = base64.b64decode(
            payload.pdf_base64,
            validate=True,
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF Base64 data.",
        )

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="PDF file is empty.",
        )

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
    # GEMINI REST REQUEST
    # --------------------------------------------------------

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
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": payload.pdf_base64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
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

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:
        response = await fetch(endpoint, fetch_options)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini connection failed: {exc}",
        )

    try:
        status_code = int(response.status)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Gemini HTTP status: {exc}",
        )

    try:
        response_text = str(await response.text())
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read Gemini response body: {exc}",
        )

    if status_code < 200 or status_code >= 300:
        try:
            error_data = json.loads(response_text)
            error_message = error_data.get("error", {}).get(
                "message",
                response_text,
            )
        except Exception:
            error_message = response_text

        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error ({status_code}): {error_message}",
        )

    # --------------------------------------------------------
    # PARSE GEMINI RESPONSE
    # --------------------------------------------------------

    try:
        gemini_data = json.loads(response_text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned non-JSON response: {exc}",
        )

    try:
        candidates = gemini_data.get("candidates") or []
        if not candidates:
            raise ValueError("No candidates returned.")

        parts = candidates[0].get("content", {}).get("parts", [])
        generated_text = ""

        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                generated_text = part["text"]
                break

        if not generated_text:
            raise ValueError("No text part returned.")

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not extract Gemini result: {exc}",
        )

    try:
        result = json.loads(generated_text)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid structured JSON: {exc}",
        )

    # --------------------------------------------------------
    # NORMALIZE RESULT
    # --------------------------------------------------------

    raw_equipment = result.get("equipment", [])

    if not isinstance(raw_equipment, list):
        raise HTTPException(
            status_code=502,
            detail="Gemini result does not contain an equipment array.",
        )

    equipment = [
        normalize_equipment(item)
        for item in raw_equipment
        if isinstance(item, dict)
    ]

    return {
        "status": "ok",
        "filename": payload.filename,
        "extractor_version": "0.2.0",
        "equipment_count": len(equipment),
        "equipment": equipment,
    }


# ============================================================
# CLOUDFLARE PYTHON WORKER ENTRYPOINT
# ============================================================

Default = asgi.entrypoint(app)
