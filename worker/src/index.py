"""
PDF Equipment Extractor - V1
Cloudflare Python Worker + FastAPI + Gemini REST API.

V1 scope:
- Receive one PDF as base64 JSON.
- Send the PDF directly to Gemini's native PDF/document understanding.
- Detect multiple equipment items in the same PDF.
- Extract the Excel-oriented parameters agreed for the project.
- Normalize missing values to "-".
- Return structured JSON for frontend review.

No permanent PDF storage and no Excel writing are included yet.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from workers import asgi

app = FastAPI(
    title="PDF Equipment Extractor",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
)

# V1 frontend is hosted on GitHub Pages, so cross-origin API calls are required.
# For production, replace "*" with your exact GitHub Pages origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_PDF_BYTES = 15 * 1024 * 1024
DEFAULT_MODEL = "gemini-3.5-flash-lite"

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
    filename: str = Field(default="document.pdf")
    mime_type: str = Field(default="application/pdf")
    pdf_base64: str


def equipment_schema() -> dict[str, Any]:
    properties = {
        key: {
            "type": "string",
            "description": f"Value for {key}. Return '-' when the requested value is not visible or cannot be found."
        }
        for key in PARAMETER_KEYS
    }

    return {
        "type": "object",
        "properties": {
            "equipment": {
                "type": "array",
                "description": "Every equipment item detected in the PDF drawing.",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": PARAMETER_KEYS,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["equipment"],
        "additionalProperties": False,
    }


EXTRACTION_PROMPT = """
You are an engineering document extraction system.

TASK
Analyze the attached engineering PDF/P&ID and identify EVERY equipment item that is relevant
to the equipment datasheet extraction task.

The PDF may contain many valves, instruments, piping lines, fittings and symbols.
DO NOT count those as equipment unless they are clearly an equipment item with an equipment
tag/identification and equipment description.

IMPORTANT
- One PDF can contain multiple equipment items.
- Detect all equipment items in the document, not only the first one.
- Use the equipment tag/header as the primary anchor.
- Read the surrounding equipment data block and connected labels when they clearly belong
  to that equipment.
- Do not invent, estimate or infer engineering values.
- Preserve the engineering value as shown in the source when possible.
- If a requested parameter is not present, return "-".
- If a field is not applicable to that equipment (for example tube-side data on a vessel),
  return "-".
- Do not convert units unless the source itself gives the converted value.
- Do not use information from another equipment to fill a missing field.
- For shell/tube equipment, keep shell-side and tube-side values separate.
- For a vessel/drum, put its pressure/temperature under shell-side fields and tube-side
  fields as "-".
- "DUTY" or other source information that does not have a dedicated target field may be
  retained in "remarks".
- The requested target fields are based on the user's Excel template.

TARGET FIELDS
1. tag_no
2. equipment_name
3. type
4. diameter
5. diameter_od_id
6. length
7. length_remarks
8. height
9. insulation
10. insulation_size
11. insulation_type
12. shell_pressure
13. shell_min_temp
14. shell_max_temp
15. tube_pressure
16. tube_min_temp
17. tube_max_temp
18. service_type
19. service_description
20. remarks

OUTPUT
Return JSON matching the supplied schema. Do not return commentary outside the JSON.

QUALITY CHECK
Before returning the result, verify that:
- multiple equipment items were not merged into one;
- instruments/valves were not incorrectly counted as equipment;
- missing values are "-";
- tag and equipment name belong to the same equipment;
- shell and tube data are not mixed;
- no engineering value was invented.
"""


def normalize_value(value: Any) -> str:
    if value is None:
        return "-"
    value = str(value).strip()
    if not value:
        return "-"
    if value.lower() in {"null", "none", "n/a", "na", "not found", "unknown"}:
        return "-"
    return value


def normalize_result(data: Any) -> dict[str, list[dict[str, str]]]:
    if not isinstance(data, dict):
        return {"equipment": []}

    raw_items = data.get("equipment", [])
    if not isinstance(raw_items, list):
        raw_items = []

    normalized = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        row = {key: normalize_value(item.get(key)) for key in PARAMETER_KEYS}

        # A result without a tag and name is generally not a usable equipment record.
        # We keep it only when at least one of the two is present.
        if row["tag_no"] == "-" and row["equipment_name"] == "-":
            continue

        normalized.append(row)

    return {"equipment": normalized}


def parse_gemini_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()

    # Structured output should already be valid JSON. This fallback handles accidental
    # markdown fences without weakening the extraction rules.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid JSON: {exc}",
        ) from exc


async def call_gemini(pdf_bytes: bytes, mime_type: str, env: Any) -> dict[str, Any]:
    api_key = getattr(env, "GEMINI_API_KEY", None)
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured in the Cloudflare Worker secret.",
        )

    model = getattr(env, "GEMINI_MODEL", None) or DEFAULT_MODEL

    # Use the Workers runtime fetch API through the Python FFI.
    from js import fetch  # type: ignore

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(pdf_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": equipment_schema(),
        },
    }

    response = await fetch(
        endpoint,
        {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": json.dumps(request_body),
        },
    )

    status = int(response.status)
    response_text = await response.text()

    if status < 200 or status >= 300:
        # Avoid returning the API key. The Gemini response itself is safe to expose
        # to the frontend as an error message, but trim it to keep UI errors manageable.
        detail = response_text[:1200]
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error ({status}): {detail}",
        )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini API returned a non-JSON response.",
        ) from exc

    candidates = payload.get("candidates") or []
    if not candidates:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no candidates.",
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    model_text = ""
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            model_text += part["text"]

    if not model_text:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no extraction text.",
        )

    return parse_gemini_json(model_text)


@app.get("/")
async def root():
    return {
        "name": "PDF Equipment Extractor",
        "version": "0.1.0",
        "status": "ok",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "pdf_direct_vision": True,
    }


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Any):
    if payload.mime_type.lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    if not payload.pdf_base64:
        raise HTTPException(status_code=400, detail="PDF data is missing.")

    try:
        pdf_bytes = base64.b64decode(payload.pdf_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data.") from exc

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded data is not a valid PDF.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the V1 limit of {MAX_PDF_BYTES // (1024 * 1024)} MB.",
        )

    env = request.scope["env"]

    result = await call_gemini(pdf_bytes, payload.mime_type, env)
    normalized = normalize_result(result)

    return {
        "status": "ok",
        "filename": payload.filename,
        "equipment_count": len(normalized["equipment"]),
        "equipment": normalized["equipment"],
    }


Default = asgi.entrypoint(app)
