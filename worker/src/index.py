from typing import Any
import asyncio
import base64
import json
import random
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from workers import WorkerEntrypoint, Response, asgi
from js import Object, fetch
from pyodide.ffi import to_js

APP_VERSION = "1.0.0"
PRIMARY_MODEL_DEFAULT = "gemini-3.8-flash"
FALLBACK_MODEL_DEFAULT = "gemini-3.7-flash"
LAST_RESORT_MODEL_DEFAULT = "gemini-3.5-flash-lite"
MAX_IMAGE_B64 = 6 * 1024 * 1024
MAX_TILES = 8
MAX_TAGS = 300

app = FastAPI(title="Piping Tag Extractor", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class Tile(BaseModel):
    id: str
    mime_type: str = "image/jpeg"
    image_base64: str

class AnalyzeRequest(BaseModel):
    filename: str
    pages: list[dict[str, Any]] = Field(min_length=1, max_length=5)

TAG_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "drawing_no": {"type": "STRING"},
        "tags": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "tag_no": {"type": "STRING"},
                    "size": {"type": "STRING"},
                    "tile_id": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                },
                "required": ["tag_no", "size", "tile_id", "evidence"],
            },
        },
    },
    "required": ["drawing_no", "tags"],
}

VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "drawing_no": {"type": "STRING"},
        "tags": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "tag_no": {"type": "STRING"},
                    "size": {"type": "STRING"},
                    "status": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["tag_no", "size", "status", "reason"],
            },
        },
    },
    "required": ["drawing_no", "tags"],
}

PROMPT_DISCOVER = r'''
You are an expert engineering P&ID line-number extraction system.

The images below come from ONE P&ID sheet. The goal is to extract PIPING / LINE TAGS only.
Do NOT extract equipment tags, valve tags, instrument tags, dimensions, elevations, notes, reference drawings, or text that is not a piping/line identification.

The drawing is raster/visual. Read the actual characters in the images. Do not rely on assumptions.

IMPORTANT TILE RULE:
- Inspect EVERY TILE independently.
- The full-page image is only for context; the tiles are the high-detail source.
- A tag may be small and near a pipe line.
- Do not miss tags near page edges or in the lower/title-block areas.
- The same tag may appear more than once; report it once after checking duplicates.

WHAT COUNTS AS A PIPING TAG:
A line/piping identification printed along or immediately beside a process/utility pipe, generally containing a line/service/area sequence and often an inch size such as 1", 2", 3", etc. Copy the complete visible tag exactly.

DO NOT use the sample Excel values as data for this PDF. The Excel is only a format reference.

SIZE RULE:
- If the tag itself contains an inch size such as 1", 2", 1/2", 10", use that size.
- If a separate line-size label is explicitly associated with the same piping tag, use it.
- Do NOT use nearby valve size/nozzle size unless it is clearly the line size for that tag.
- If size cannot be proven, return "-".

DRAWING NUMBER:
Read the drawing number from the title block if clearly visible. It is common to all rows.
Do not confuse project number, revision, file name, or job number with drawing number.

EVIDENCE:
For each tag, briefly state where/how it is visible, e.g. "printed directly on horizontal process line in tile R1C2".

Return JSON only using the supplied schema.
'''

PROMPT_VERIFY = r'''
You are the verification stage for an engineering P&ID piping-tag extraction.

You receive the same page images and a preliminary list of candidate piping tags.
Independently inspect the relevant image areas and verify every candidate.

KEEP a candidate ONLY if it is genuinely a PIPING/LINE TAG printed on or immediately associated with a pipe/line.
REMOVE candidates that are:
- equipment tags/numbers;
- valve tags or valve identifiers;
- instrument tags;
- nozzle numbers;
- dimensions/elevations;
- reference drawing numbers;
- notes or revision information;
- text that is not a line identification.

COPY THE TAG EXACTLY as visually shown. Do not normalize its engineering meaning or invent missing characters.

SIZE:
Prefer the inch size visibly embedded in the tag. For example, 1HC106-1"-FC2L -> size 1; 1HC1-2"-FG2D -> size 2.
If a quoted inch size is not visibly present and no explicit associated line size exists, return "-".

DUPLICATES:
If the same tag appears multiple times, return it only once.

DRAWING NUMBER:
Verify it from the title block. For this sheet it may look like D1D-A-25; do not assume it unless visually confirmed.

Return only verified tags. If a candidate is uncertain, remove it rather than guessing.
Return JSON only.
'''


def env_value(env: Any, name: str, default: str) -> str:
    try:
        value = getattr(env, name)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return default


def normalize_text(value: Any) -> str:
    if value is None:
        return "-"
    s = str(value).strip()
    s = s.replace("“", '"').replace("”", '"').replace("″", '"')
    s = re.sub(r"\s+", " ", s)
    return s or "-"


def size_from_tag(tag: str) -> str | None:
    if not tag or tag == "-":
        return None
    m = re.search(r'(?:^|[-_/])\s*(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*["”″]', tag)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def normalize_tags(items: Any) -> list[dict]:
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        tag = normalize_text(item.get("tag_no"))
        if tag == "-":
            continue
        key = re.sub(r"\s+", "", tag).upper()
        if key in seen:
            continue
        seen.add(key)
        parsed_size = size_from_tag(tag)
        ai_size = normalize_text(item.get("size"))
        size = parsed_size or ai_size
        out.append({
            "tag_no": tag,
            "size": size,
            "tile_id": normalize_text(item.get("tile_id")),
            "evidence": normalize_text(item.get("evidence")),
        })
        if len(out) >= MAX_TAGS:
            break
    return out


def model_config(model: str) -> dict:
    cfg = {"responseMimeType": "application/json", "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    if model.startswith("gemini-3."):
        cfg["thinkingConfig"] = {"thinkingLevel": "high"}
    return cfg

async def sleep_retry(seconds: float):
    await asyncio.sleep(seconds + random.uniform(0, 0.7))

RETRYABLE = {408, 429, 500, 502, 503, 504}
RETRY_DELAYS = [2.0, 5.0]

def error_message(text: str) -> str:
    try:
        obj = json.loads(text)
        return str(obj.get("error", {}).get("message", text))
    except Exception:
        return text or "Gemini API error"

async def gemini_call(api_key: str, model: str, prompt: str, images: list[dict], schema: dict) -> dict:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    parts = [{"text": prompt}]
    for image in images:
        parts.append({"text": f"IMAGE/TILE ID: {image['id']}"})
        parts.append({"inline_data": {"mime_type": image["mime_type"], "data": image["data"]}})
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {**model_config(model), "responseSchema": schema},
    }
    options = to_js({"method":"POST","headers":{"Content-Type":"application/json"},"body":json.dumps(body)}, dict_converter=Object.fromEntries)
    last_status = 502
    last_msg = "Gemini request failed"
    for attempt in range(len(RETRY_DELAYS)+1):
        try:
            resp = await fetch(endpoint, options)
            status = int(resp.status)
            txt = str(await resp.text())
        except Exception as exc:
            status, txt = 502, str(exc)
        if 200 <= status < 300:
            try:
                data = json.loads(txt)
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ValueError("Gemini returned no candidates")
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return json.loads(text)
            except Exception as exc:
                raise HTTPException(502, f"Gemini returned invalid structured output: {exc}")
        last_status, last_msg = status, error_message(txt)
        if status not in RETRYABLE or attempt >= len(RETRY_DELAYS):
            break
        await sleep_retry(RETRY_DELAYS[attempt])
    raise HTTPException(503 if last_status in RETRYABLE else 502, f"Gemini API error ({last_status}): {last_msg}")

async def gemini_fallbacks(api_key: str, models: list[str], prompt: str, images: list[dict], schema: dict) -> tuple[dict,str]:
    last = None
    for model in models:
        if not model:
            continue
        try:
            return await gemini_call(api_key, model, prompt, images, schema), model
        except HTTPException as exc:
            last = exc
            if exc.status_code not in {408,429,500,502,503,504}:
                raise
    if last:
        raise last
    raise HTTPException(503, "No Gemini model configured")


def response_headers(request: Request) -> dict:
    origin = request.headers.get("Origin")
    allowed = {"https://hendriseptian.github.io", "http://localhost:8787", "http://127.0.0.1:8787"}
    return {
        "Access-Control-Allow-Origin": origin if origin in allowed else "https://hendriseptian.github.io",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": request.headers.get("Access-Control-Request-Headers", "Content-Type"),
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }

@app.get("/health")
async def health(request: Request):
    env = request.scope.get("env")
    return {
        "status":"ok",
        "model":env_value(env,"GEMINI_MODEL",PRIMARY_MODEL_DEFAULT),
        "fallback_model":env_value(env,"GEMINI_FALLBACK_MODEL",FALLBACK_MODEL_DEFAULT),
        "last_resort_model":env_value(env,"GEMINI_LAST_RESORT_MODEL",LAST_RESORT_MODEL_DEFAULT),
        "pipeline":"PDF.js high-resolution tiling + multi-model vision + verification",
        "extractor_version":APP_VERSION,
        "output":"Piping Tag / P&ID No. / Size",
        "line_description":"blank",
    }

@app.post("/api/analyze-piping")
async def analyze_piping(payload: AnalyzeRequest, request: Request):
    env = request.scope.get("env")
    api_key = env_value(env, "GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY secret is not configured.")
    if not payload.pages:
        raise HTTPException(400, "No rendered PDF pages supplied.")

    primary = env_value(env,"GEMINI_MODEL",PRIMARY_MODEL_DEFAULT)
    fallback = env_value(env,"GEMINI_FALLBACK_MODEL",FALLBACK_MODEL_DEFAULT)
    last = env_value(env,"GEMINI_LAST_RESORT_MODEL",LAST_RESORT_MODEL_DEFAULT)

    all_tags = []
    drawing_numbers = []
    page_reports = []

    for page in payload.pages:
        page_no = int(page.get("page_no", 1))
        full_page = page.get("full_page") or {}
        tiles = page.get("tiles") or []
        images = []
        if full_page.get("data"):
            images.append({"id": f"P{page_no}-FULL", "mime_type": full_page.get("mime_type","image/jpeg"), "data": full_page["data"]})
        for tile in tiles[:MAX_TILES]:
            if tile.get("data"):
                images.append({"id": tile.get("id", f"P{page_no}-TILE"), "mime_type": tile.get("mime_type","image/jpeg"), "data": tile["data"]})
        if not images:
            continue

        discovery, model1 = await gemini_fallbacks(api_key,[primary,fallback,last],PROMPT_DISCOVER,images,TAG_SCHEMA)
        candidates = normalize_tags(discovery.get("tags"))
        drawing = normalize_text(discovery.get("drawing_no"))
        if drawing != "-": drawing_numbers.append(drawing)

        candidate_text = json.dumps({"drawing_no": drawing, "tags": candidates}, ensure_ascii=False)
        verify_prompt = PROMPT_VERIFY + "\nPRELIMINARY CANDIDATES:\n" + candidate_text
        verified, model2 = await gemini_fallbacks(api_key,[primary,fallback,last],verify_prompt,images,VERIFY_SCHEMA)
        verified_tags = normalize_tags(verified.get("tags"))
        vdrawing = normalize_text(verified.get("drawing_no"))
        if vdrawing != "-": drawing_numbers.append(vdrawing)
        all_tags.extend(verified_tags)
        page_reports.append({"page":page_no,"discovery_model":model1,"verification_model":model2,"candidate_count":len(candidates),"verified_count":len(verified_tags)})

    # Global dedupe + deterministic size extraction from the exact tag string.
    final = []
    seen = set()
    for item in all_tags:
        tag = normalize_text(item.get("tag_no"))
        key = re.sub(r"\s+", "", tag).upper()
        if key in seen or tag == "-":
            continue
        seen.add(key)
        parsed = size_from_tag(tag)
        size = parsed or normalize_text(item.get("size"))
        final.append({"tag_no":tag,"pid_no":"-","from":"","to":"","size":size,"evidence":normalize_text(item.get("evidence"))})
        if len(final) >= MAX_TAGS:
            break

    pid = "-"
    # Prefer a drawing number that looks like the title-block drawing number; otherwise first verified value.
    for candidate in drawing_numbers:
        if candidate != "-":
            pid = candidate
            break
    for row in final:
        row["pid_no"] = pid

    final.sort(key=lambda x: x["tag_no"])
    return {
        "status":"ok",
        "filename":payload.filename,
        "drawing_no":pid,
        "count":len(final),
        "piping_tags":final,
        "pages":page_reports,
        "line_description_blank":True,
    }

class Default(WorkerEntrypoint):
    async def fetch(self, request):
        if request.method == "OPTIONS":
            h = response_headers(request)
            return Response("", status=204, headers=h)
        try:
            resp = await asgi.fetch(app, request, self.env)
            h = response_headers(request)
            for k,v in h.items():
                resp.headers.set(k,v)
            return resp
        except Exception as exc:
            return Response(json.dumps({"detail":str(exc)}), status=500, headers={**response_headers(request),"Content-Type":"application/json"})
