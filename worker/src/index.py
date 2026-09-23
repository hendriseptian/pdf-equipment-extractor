from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import base64
import json

from workers import asgi
from js import fetch


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PDF Equipment Extractor",
    version="0.1.0",
    description="Extract equipment data from engineering PDF drawings using Gemini."
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIG
# ============================================================

MAX_PDF_SIZE = 15 * 1024 * 1024  # 15 MB


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
# NORMALIZE VALUE
# ============================================================

def normalize_value(value: Any) -> str:
    """
    Convert missing / empty / null-like values to '-'.
    Everything else is converted to string.
    """

    if value is None:
        return "-"

    text = str(value).strip()

    if not text:
        return "-"

    null_values = {
        "null",
        "none",
        "n/a",
        "na",
        "not available",
        "not found",
        "unknown",
        "undefined",
    }

    if text.lower() in null_values:
        return "-"

    return text


# ============================================================
# NORMALIZE EQUIPMENT
# ============================================================

def normalize_equipment(equipment: dict) -> dict:
    """
    Make sure every expected parameter exists.
    Missing values become '-'.
    """

    result = {}

    for key in PARAMETER_KEYS:
        result[key] = normalize_value(equipment.get(key))

    return result


# ============================================================
# GEMINI RESPONSE SCHEMA
# ============================================================

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "equipment": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "tag_no": {
                        "type": "STRING"
                    },
                    "equipment_name": {
                        "type": "STRING"
                    },
                    "type": {
                        "type": "STRING"
                    },

                    "diameter": {
                        "type": "STRING"
                    },
                    "diameter_od_id": {
                        "type": "STRING"
                    },
                    "length": {
                        "type": "STRING"
                    },
                    "length_remarks": {
                        "type": "STRING"
                    },
                    "height": {
                        "type": "STRING"
                    },

                    "insulation": {
                        "type": "STRING"
                    },
                    "insulation_size": {
                        "type": "STRING"
                    },
                    "insulation_type": {
                        "type": "STRING"
                    },

                    "shell_pressure": {
                        "type": "STRING"
                    },
                    "shell_min_temp": {
                        "type": "STRING"
                    },
                    "shell_max_temp": {
                        "type": "STRING"
                    },

                    "tube_pressure": {
                        "type": "STRING"
                    },
                    "tube_min_temp": {
                        "type": "STRING"
                    },
                    "tube_max_temp": {
                        "type": "STRING"
                    },

                    "service_type": {
                        "type": "STRING"
                    },
                    "service_description": {
                        "type": "STRING"
                    },

                    "remarks": {
                        "type": "STRING"
                    },
                },
                "required": PARAMETER_KEYS,
            },
        }
    },
    "required": ["equipment"],
}


# ============================================================
# EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = """
You are an engineering drawing data extraction AI.

Analyze the provided PDF engineering drawing carefully.

The PDF may contain ONE or MULTIPLE equipment items.

Your task is to identify EVERY relevant equipment item shown in the drawing
and extract the available engineering parameters.

IMPORTANT RULES:

1. Detect ALL equipment in the PDF.
   Do not stop after finding the first equipment.

2. Equipment means identifiable process/mechanical equipment such as:
   - Vessel
   - Separator
   - Drum
   - Tank
   - Heat exchanger
   - Cooler
   - Heater
   - Pump
   - Compressor
   - Column
   - Reactor
   - Other clearly identifiable major equipment.

3. DO NOT count these as equipment:
   - Valves
   - Instruments
   - Piping
   - Fittings
   - Flanges
   - Reducers
   - Elbows
   - Small inline components
   unless they are clearly identified as a major equipment item.

4. Do NOT invent, estimate, calculate, or guess values.

5. Only extract information that is actually visible or explicitly stated
   in the PDF.

6. If information is missing, unclear, or not applicable:
   return "-"

7. Preserve engineering values as written in the drawing whenever possible.

8. Keep SHELL SIDE and TUBE SIDE information separate.

9. For vessels, drums, tanks, separators, columns, etc.:
   - Put vessel design pressure in shell_pressure.
   - Put vessel design temperature in shell_min_temp or shell_max_temp
     only when the drawing explicitly identifies the temperature.
   - Do not invent a minimum temperature if only one design temperature exists.
   - Tube-side fields should be "-".

10. For heat exchangers/coolers:
    - Keep shell-side and tube-side pressure/temperature separate.
    - Do not mix the two sides.

11. Diameter:
    Extract the diameter exactly as shown.

12. Diameter OD/ID:
    If the drawing explicitly states OD or ID, record it.
    Otherwise return "-".

13. Length:
    Extract vessel/equipment length where explicitly shown.

14. Height:
    Extract height where explicitly shown.

15. Insulation:
    If the drawing explicitly says NONE, return "No".
    If insulation is explicitly indicated, return "Yes".
    If not stated, return "-".

16. Service:
    Use the explicitly stated service/fluid information.
    Do not guess the fluid based only on the equipment name.

17. DUTY:
    If a DUTY value is shown but there is no dedicated field,
    preserve it in remarks.

18. Tag number:
    Extract the equipment tag exactly as shown.

19. Equipment name:
    Extract the equipment description/name exactly or very close to
    the wording shown in the drawing.

20. TYPE:
    Identify the equipment type only when it is clear from the drawing.
    Otherwise return "-".

21. Return JSON only according to the requested schema.

22. Every equipment object MUST contain every requested field.

23. Use "-" for every unavailable field.

Do not provide explanations outside the JSON.
"""


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "PDF Equipment Extractor",
        "status": "ok",
        "version": "0.1.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health(request: Request):
    env = request.scope.get("env")

    if env is None:
        model = "unknown"
    else:
        model = getattr(
            env,
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )

    return {
        "status": "ok",
        "model": model,
        "pdf_direct_vision": True
    }


# ============================================================
# ANALYZE PDF
# ============================================================

@app.post("/api/analyze")
async def analyze(
    payload: AnalyzeRequest,
    request: Request
):
    """
    Receive PDF as Base64,
    send directly to Gemini,
    return structured equipment JSON.
    """

    # --------------------------------------------------------
    # GET CLOUDFLARE ENV
    # --------------------------------------------------------

    env = request.scope.get("env")

    if env is None:
        raise HTTPException(
            status_code=500,
            detail="Cloudflare environment is not available."
        )

    api_key = getattr(env, "GEMINI_API_KEY", None)

    model = getattr(
        env,
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    )

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # VALIDATE MIME TYPE
    # --------------------------------------------------------

    if payload.mime_type.lower() != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only application/pdf is supported."
        )

    # --------------------------------------------------------
    # DECODE PDF
    # --------------------------------------------------------

    try:
        pdf_bytes = base64.b64decode(
            payload.pdf_base64,
            validate=True
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF Base64 data."
        )

    # --------------------------------------------------------
    # SIZE CHECK
    # --------------------------------------------------------

    pdf_size = len(pdf_bytes)

    if pdf_size == 0:
        raise HTTPException(
            status_code=400,
            detail="PDF file is empty."
        )

    if pdf_size > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF file is too large. Maximum size is 15 MB."
        )

    # --------------------------------------------------------
    # PDF SIGNATURE CHECK
    # --------------------------------------------------------

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file does not appear to be a valid PDF."
        )

    # --------------------------------------------------------
    # GEMINI ENDPOINT
    # --------------------------------------------------------

    endpoint = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/"
        f"{model}:generateContent"
        f"?key={api_key}"
    )

    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": EXTRACTION_PROMPT
                    },
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": payload.pdf_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA
        }
    }

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:
        response = await fetch(
            endpoint,
            {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps(request_body)
            }
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Gemini: {str(exc)}"
        )

    # --------------------------------------------------------
    # READ GEMINI RESPONSE
    # --------------------------------------------------------

    try:
        response_text = await response.text()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to read Gemini response: {str(exc)}"
        )

    # --------------------------------------------------------
    # GEMINI HTTP ERROR
    # --------------------------------------------------------

    if response.status < 200 or response.status >= 300:

        try:
            error_data = json.loads(response_text)

            error_message = (
                error_data
                .get("error", {})
                .get("message", response_text)
            )

        except Exception:
            error_message = response_text

        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {error_message}"
        )

    # --------------------------------------------------------
    # PARSE GEMINI RESPONSE
    # --------------------------------------------------------

    try:
        gemini_data = json.loads(response_text)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid response."
        )

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    try:
        candidates = gemini_data.get("candidates", [])

        if not candidates:
            raise ValueError(
                "Gemini returned no candidates."
            )

        content = candidates[0].get("content", {})

        parts = content.get("parts", [])

        if not parts:
            raise ValueError(
                "Gemini returned no content parts."
            )

        generated_text = parts[0].get("text", "")

        if not generated_text:
            raise ValueError(
                "Gemini returned empty text."
            )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to extract Gemini result: {str(exc)}"
        )

    # --------------------------------------------------------
    # PARSE STRUCTURED JSON
    # --------------------------------------------------------

    try:
        result = json.loads(generated_text)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned invalid JSON."
        )

    # --------------------------------------------------------
    # VALIDATE EQUIPMENT ARRAY
    # --------------------------------------------------------

    equipment_list = result.get("equipment", [])

    if not isinstance(equipment_list, list):
        equipment_list = []

    # --------------------------------------------------------
    # NORMALIZE ALL EQUIPMENT
    # --------------------------------------------------------

    normalized_equipment = []

    for item in equipment_list:

        if not isinstance(item, dict):
            continue

        normalized_equipment.append(
            normalize_equipment(item)
        )

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {
        "status": "ok",
        "filename": payload.filename,
        "equipment_count": len(normalized_equipment),
        "equipment": normalized_equipment
    }


# ============================================================
# CLOUDFLARE PYTHON WORKER ENTRYPOINT
# ============================================================

Default = asgi.entrypoint(app)
