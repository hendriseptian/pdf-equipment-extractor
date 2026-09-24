from typing import Any
import base64
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from workers import WorkerEntrypoint, Response, asgi
from js import Object, fetch
from pyodide.ffi import to_js


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PDF Equipment Extractor",
    version="0.1.1",
    description="Extract equipment data from engineering PDFs using Gemini.",
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
    }:
        return "-"

    return text


def normalize_equipment(item: dict) -> dict:
    return {
        key: normalize_value(item.get(key))
        for key in PARAMETER_KEYS
    }


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
# GEMINI SCHEMA
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
# EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = """
You are an engineering drawing data extraction AI.

Analyze the entire provided PDF engineering drawing.

The PDF may contain ONE or MULTIPLE equipment items.
Identify EVERY relevant equipment item before returning the result.

EQUIPMENT RULES
1. Include clearly identifiable major/process equipment:
   vessels, separators, drums, tanks, heat exchangers, coolers,
   heaters, pumps, compressors, columns, reactors, and similar equipment.

2. Do NOT count valves, instruments, piping, fittings, flanges,
   reducers, elbows, or small inline components as equipment.

3. Do not invent, estimate, calculate, or guess information.

4. Use only information actually visible or explicitly stated in the PDF.

5. If a value is missing, unclear, or not applicable, return "-".

6. Preserve engineering values and units as shown whenever possible.

FIELD RULES
7. tag_no:
   Extract the equipment tag exactly as shown.

8. equipment_name:
   Extract the equipment description/name from the drawing.

9. type:
   Identify the equipment type only when clearly supported by the drawing.

10. diameter:
    Extract explicitly shown diameter.

11. diameter_od_id:
    Record OD or ID only when explicitly stated.
    Otherwise "-".

12. length:
    Extract explicitly shown equipment/vessel length.

13. length_remarks:
    Preserve useful length notation such as T-T, T/T, etc.

14. height:
    Extract explicitly shown height.

15. insulation:
    If explicitly "NONE", return "No".
    If insulation is explicitly indicated, return "Yes".
    If not stated, return "-".

16. insulation_size and insulation_type:
    Extract only when explicitly shown.

17. SHELL SIDE:
    Keep shell_pressure, shell_min_temp, and shell_max_temp separate
    from tube-side values.

18. TUBE SIDE:
    Keep tube_pressure, tube_min_temp, and tube_max_temp separate.

19. For vessels, drums, separators, tanks, columns, etc.:
    Put explicitly stated vessel design pressure in shell_pressure.
    Put a stated design temperature in shell_min_temp or shell_max_temp
    only according to what is actually stated.
    Do not invent a minimum temperature.
    Tube-side fields are "-".

20. For heat exchangers/coolers:
    Do not mix shell-side and tube-side values.

21. service_type:
    Use L, V, G, or another value only when explicitly supported.

22. service_description:
    Extract explicitly stated service/fluid description.
    Do not guess from the equipment name.

23. DUTY:
    If DUTY is shown but there is no dedicated field,
    preserve the DUTY information in remarks.

24. Every equipment object must contain every requested field.

25. Return JSON only according to the supplied schema.
"""


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "PDF Equipment Extractor",
        "status": "ok",
        "version": "0.1.1",
    }


@app.get("/health")
async def health(request: Request):
    env = get_env(request)
    model = get_binding(env, "GEMINI_MODEL", "gemini-3.5-flash-lite")

    return {
        "status": "ok",
        "model": str(model),
        "pdf_direct_vision": True,
        "extractor_version": "0.6.0",
    }


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request):
    # --------------------------------------------------------
    # ENV / SECRET
    # --------------------------------------------------------

    env = get_env(request)

    api_key = get_binding(env, "GEMINI_API_KEY")
    model = get_binding(env, "GEMINI_MODEL", "gemini-3.5-flash-lite")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    model = str(model).strip()

    if not model:
        model = "gemini-3.5-flash-lite"

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
    # GEMINI REQUEST
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

    # Cloudflare Python Workers uses Pyodide FFI for JavaScript APIs.
    # Explicitly convert the Python options object to a JS object.
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

    # --------------------------------------------------------
    # READ RESPONSE
    # --------------------------------------------------------

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
            error_message = (
                error_data.get("error", {}).get(
                    "message",
                    response_text,
                )
            )
        except Exception:
            error_message = response_text

        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error ({status_code}): {error_message}",
        )

    # --------------------------------------------------------
    # PARSE GEMINI JSON
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
        "equipment_count": len(equipment),
        "equipment": equipment,
    }


# ============================================================
# CLOUDFLARE PYTHON WORKER ENTRYPOINT
# ============================================================
#
# CORS is handled OUTSIDE FastAPI at the Worker boundary.
# This is important because the browser request originates
# from GitHub Pages while the API is on workers.dev.
#
# The frontend sends application/json, so the browser performs
# an OPTIONS preflight before POST /api/analyze.
# ============================================================

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        origin = request.headers.get("Origin")

        # Allow the current GitHub Pages frontend.
        # Keep localhost origins for future local testing.
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

        # ----------------------------------------------------
        # CORS PREFLIGHT
        # ----------------------------------------------------
        if request.method == "OPTIONS":
            requested_headers = request.headers.get(
                "Access-Control-Request-Headers"
            )

            if requested_headers:
                cors_headers["Access-Control-Allow-Headers"] = (
                    requested_headers
                )

            requested_method = request.headers.get(
                "Access-Control-Request-Method"
            )

            if requested_method:
                cors_headers["Access-Control-Allow-Methods"] = (
                    requested_method
                ) + ", OPTIONS"

            return Response(
                "",
                status=204,
                headers=cors_headers,
            )

        # ----------------------------------------------------
        # FASTAPI / ASGI
        # ----------------------------------------------------
        try:
            response = await asgi.fetch(
                app,
                request,
                self.env,
            )

            # Cloudflare Workers allows response headers to be
            # modified on the response object. This avoids
            # rebuilding the response body/stream.
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
            response.headers.set(
                "Vary",
                "Origin",
            )

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
