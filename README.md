# P&ID Piping Tag Extractor V1

Purpose: scan P&ID PDF pages and extract piping/line tags into the supplied Excel template.

## Pipeline
1. Browser renders each PDF page with PDF.js at high resolution.
2. Each page is split into 6 overlapping visual tiles (3x2) plus a full-page context image.
3. Gemini 3.8 Flash inspects every tile and extracts candidate piping tags.
4. Gemini verifies the candidates using the same visual tiles.
5. Tags are deduplicated; NPS is deterministically read from the exact quoted inch value in the tag when present.
6. Preview is editable in the browser.
7. `template/piping.xlsx` is used for Excel export; From/To remain blank.

## Cloudflare
Secret required: `GEMINI_API_KEY`.
Variables are in `worker/wrangler.jsonc`.

## Deploy
Use the same Cloudflare Python Workers deployment method already working in the existing project:
`uv run pywrangler deploy`


Deployment note: this project uses Cloudflare Pywrangler via the `workers-py` dev dependency. Deploy with `uv run pywrangler deploy`.
