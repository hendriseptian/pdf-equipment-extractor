from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Any
import base64
import json

from workers import asgi
from js import fetch

app = FastAPI(
    title="PDF Equipment Extractor",
    version="0.4.0",
    description="Extract equipment data from engineering PDF drawings using Gemini."
)

@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    # Explicit CORS handling so browser requests always receive the
    # Access-Control-Allow-Origin header, including error responses.
    if request.method == "OPTIONS":
        response = Response(status_code=204)
    else:
        try:
            response = await call_next(request)
        except Exception as exc:
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Worker internal error.",
                    "error": str(exc),
                },
            )

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "86400"

    return response

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

class AnalyzeRequest(BaseModel):
    filename: str
    mime_type: str
    pdf_base64: str

def normalize_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    if text.lower() in {
        "null", "none", "n/a", "na", "not available",
        "not found", "unknown", "undefined", "not specified"
    }:
        return "-"
    return text

def normalize_equipment(item: dict) -> dict:
    return {key: normalize_value(item.get(key)) for key in PARAMETER_KEYS}

# Important: the model is instructed to extract only explicit evidence.
# It must not use generic P&ID knowledge to fill engineering values.
EXTRACTION_PROMPT = r"""
You are a STRICT engineering drawing transcription engine.

Read the ENTIRE PDF visually. Identify every MAJOR EQUIPMENT item and
transcribe only values that are explicitly printed in the PDF.

THIS IS NOT AN ESTIMATION TASK.
DO NOT calculate, infer, interpolate, substitute, or borrow values from
another equipment item.

============================================================
A. EQUIPMENT IDENTIFICATION
============================================================

An equipment item must have its own identifiable equipment tag and/or
equipment description.

For every equipment item:
1. Find its exact equipment tag.
2. Find its exact equipment name/description.
3. Associate specifications ONLY with that same equipment.

Do not use a nearby pipe number, line number, valve tag, instrument tag,
or another equipment tag.

Do not count valves, instruments, piping, fittings, or flanges as equipment.

If a tag is not clearly readable, return "-" rather than inventing one.

============================================================
B. VERY IMPORTANT: FOLLOW THE EQUIPMENT'S OWN DATA BLOCK
============================================================

When an equipment has a specification/data block, use that block as the
PRIMARY SOURCE for pressure, temperature, dimensions, insulation, and duty.

Do NOT take a number simply because it is visually close on the drawing.

For each value ask:
"Is this value explicitly associated with THIS equipment?"

If the answer is not clearly yes, return "-".

============================================================
C. DIMENSIONS — STRICT RULE
============================================================

Only fill diameter/length/height when the dimension is explicitly the
equipment dimension.

NEVER use:
- pipe diameter
- pipe size
- line size
- nozzle size
- valve size
- instrument connection size
- nearby piping dimensions

as the equipment diameter.

Examples of valid equipment dimensions:
- 10'-0" Ø
- 10'-0" DIA
- 10'-0" Ø x 9'-0" T-T
- 10'-0" Ø x 40'-6" T/T

If an equipment dimension is not explicitly shown, return "-".

For a dimension written like:
10'-0" Ø × 9'-0" T-T

extract:
diameter = "10'-0""
length = "9'-0" T-T"

For:
10'-0" Ø × 40'-6" T/T

extract:
diameter = "10'-0""
length = "40'-6" T/T"

Do NOT convert units.

============================================================
D. TEMPERATURE — STRICT RULE
============================================================

Only use a temperature explicitly associated with that equipment.

If the drawing says:
DESIGN TEMP 200°F (93.3°C)

record the actual displayed design temperature.

Do not copy a temperature from another equipment.

Do not assume every equipment has 200°F.

If only one design temperature is explicitly given and the drawing does
not identify it as MIN or MAX, put it in shell_max_temp and keep
shell_min_temp as "-".

For a vessel/separator/drum with no tube side:
tube_min_temp = "-"
tube_max_temp = "-"
tube_pressure = "-"

============================================================
E. PRESSURE — STRICT RULE
============================================================

Only use a pressure explicitly associated with that equipment.

If an equipment data block has:
SHELL DESIGN PRESSURE = 700 PSIG
TUBE DESIGN PRESSURE = 125 PSIG

then:
shell_pressure = "700 PSIG"
tube_pressure = "125 PSIG"

Do not substitute a nearby line pressure.

Do not infer pressure from equipment type.

============================================================
F. HEAT EXCHANGER / COOLER
============================================================

For a cooler/heat exchanger, carefully inspect the equipment's own
SHELL and TUBE data.

Keep shell and tube information completely separate.

If the PDF explicitly shows:

SHELL:
Design Temp = 200°F (93.3°C)
Design Pressure = 700 PSIG (49.21 KG/CM2G)

TUBE:
Design Temp = 200°F (93.3°C)
Design Pressure = 125 PSIG (8.79 KG/CM2G)

then transcribe exactly those values into the corresponding fields.

Do not replace them with piping values.

============================================================
G. VESSELS / SEPARATORS / DRUMS
============================================================

For a vessel, separator, drum, tank, etc.:

- shell_pressure = its explicitly stated vessel design pressure
- shell_max_temp = its explicitly stated design temperature if only one
  temperature is provided
- tube_pressure = "-"
- tube_min_temp = "-"
- tube_max_temp = "-"

Do not invent tube-side values.

============================================================
H. INSULATION
============================================================

Only use explicit insulation information.

If the equipment block says:
INSULATION NONE

then:
insulation = "No"

If it explicitly says insulated, then:
insulation = "Yes"

Otherwise:
insulation = "-"

============================================================
I. DUTY
============================================================

If an explicit DUTY is shown, preserve it in remarks if there is no
dedicated DUTY field.

Do not calculate duty.

============================================================
J. SERVICE
============================================================

Use only explicit service/fluid information.
Do not infer fluid from equipment name alone.

============================================================
K. NO CROSS-EQUIPMENT CONTAMINATION
============================================================

This is critical.

Before assigning each field, verify that the field belongs to the SAME
equipment tag.

Never copy a value from equipment A to equipment B.

For example, if equipment A has 200°F and equipment B has 160°F,
equipment B MUST remain 160°F.

============================================================
L. OUTPUT
============================================================

Return JSON only.

Return every equipment item exactly once.

Every equipment object MUST contain all fields in the requested schema.

For every unavailable, unreadable, or not-applicable value return "-".

Do not add explanatory prose outside JSON.
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "equipment": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {key: {"type": "STRING"} for key in PARAMETER_KEYS},
                "required": PARAMETER_KEYS,
            },
        }
    },
    "required": ["equipment"],
}

@app.get("/")
async def root():
    return {
        "service": "PDF Equipment Extractor",
        "status": "ok",
        "version": "0.4.0",
    }

@app.get("/health")
async def health(request: Request):
    env = request.scope.get("env")
    model = getattr(env, "GEMINI_MODEL", "gemini-3.5-flash-lite") if env else "unknown"
    return {
        "status": "ok",
        "model": model,
        "pdf_direct_vision": True,
        "extractor_version": "0.4.0",
    }

@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request):
    env = request.scope.get("env")
    if env is None:
        raise HTTPException(500, "Cloudflare environment is not available.")

    api_key = getattr(env, "GEMINI_API_KEY", None)
    model = getattr(env, "GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY is not configured.")

    if payload.mime_type.lower() != "application/pdf":
        raise HTTPException(400, "Only application/pdf is supported.")

    try:
        pdf_bytes = base64.b64decode(payload.pdf_base64, validate=True)
    except Exception:
        raise HTTPException(400, "Invalid PDF Base64 data.")

    if not pdf_bytes:
        raise HTTPException(400, "PDF file is empty.")
    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(413, "PDF file is too large. Maximum size is 15 MB.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(400, "Uploaded file does not appear to be a valid PDF.")

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    request_body = {
        "contents": [{
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
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
        },
    }

    try:
        response = await fetch(
            endpoint,
            {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(request_body),
            },
        )
    except Exception as exc:
        raise HTTPException(502, f"Failed to connect to Gemini: {str(exc)}")

    try:
        response_text = await response.text()
    except Exception as exc:
        raise HTTPException(502, f"Failed to read Gemini response: {str(exc)}")

    if response.status < 200 or response.status >= 300:
        try:
            error_data = json.loads(response_text)
            message = error_data.get("error", {}).get("message", response_text)
        except Exception:
            message = response_text
        raise HTTPException(502, f"Gemini API error: {message}")

    try:
        gemini_data = json.loads(response_text)
        candidates = gemini_data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Gemini returned no content parts.")
        generated_text = parts[0].get("text", "")
        if not generated_text:
            raise ValueError("Gemini returned empty text.")
        result = json.loads(generated_text)
    except Exception as exc:
        raise HTTPException(502, f"Invalid Gemini structured response: {str(exc)}")

    equipment = result.get("equipment", [])
    if not isinstance(equipment, list):
        equipment = []

    normalized = [
        normalize_equipment(item)
        for item in equipment
        if isinstance(item, dict)
    ]

    return {
        "status": "ok",
        "filename": payload.filename,
        "equipment_count": len(normalized),
        "equipment": normalized,
    }

Default = asgi.entrypoint(app)
