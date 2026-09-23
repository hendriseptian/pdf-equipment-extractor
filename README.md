# PDF Equipment Extractor

V1 online prototype for extracting engineering equipment data from scanned/vector P&ID PDFs.

## Current V1 flow

```text
GitHub Pages
    |
    | PDF
    v
Cloudflare Python Worker + FastAPI
    |
    v
Gemini 3.5 Flash-Lite
    |
    | native PDF/document understanding
    v
Structured JSON
    |
    v
GitHub Pages preview
```

### Important V1 behavior

- One PDF can contain multiple equipment items.
- The AI is instructed to distinguish equipment from valves, instruments, piping and fittings.
- Requested parameters are based on `template/contoh.xlsx`.
- If a requested value is not found, the backend normalizes it to `-`.
- The AI is instructed not to invent engineering values.
- Excel writing is intentionally not included yet. V1 first validates detection and extraction accuracy.

## Project structure

```text
pdf-equipment-extractor/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── worker/
│   ├── src/
│   │   └── index.py
│   ├── pyproject.toml
│   └── wrangler.jsonc
├── template/
│   └── contoh.xlsx
└── README.md
```

## 1. Create the Gemini API key

Create a Gemini API key in Google AI Studio.

Do NOT put the key in `frontend/script.js` or commit it to GitHub.

The Worker reads it as the Cloudflare secret:

```text
GEMINI_API_KEY
```

## 2. Deploy the Worker

Open a terminal:

```bash
cd worker
```

Install/run with Pywrangler:

```bash
uv run pywrangler dev
```

For the first deployment:

```bash
uv run pywrangler deploy
```

Then add the Gemini secret:

```bash
uv run pywrangler secret put GEMINI_API_KEY
```

Paste your Gemini API key when prompted.

If you set the secret after the first deployment, deploy again if your local workflow requires it.

## 3. Test the Worker

Open:

```text
https://YOUR-WORKER.workers.dev/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "gemini-3.5-flash-lite",
  "pdf_direct_vision": true
}
```

## 4. Connect GitHub Pages

Open:

```text
frontend/script.js
```

Change:

```javascript
const API_BASE_URL = "https://YOUR-WORKER.workers.dev";
```

to your actual Worker URL.

Example:

```javascript
const API_BASE_URL = "https://pdf-equipment-extractor.your-subdomain.workers.dev";
```

Then upload the `frontend` files to your GitHub Pages repository.

## 5. Test

1. Open the GitHub Pages website.
2. Select `sheet 2.pdf` or another PDF.
3. Click **Analyze PDF**.
4. The Worker sends the PDF to Gemini.
5. Gemini detects all equipment items.
6. The web displays the extracted fields.
7. Check and edit the fields in the preview.
8. V1 currently stops at JSON preview.

## Expected behavior for the supplied sample

The supplied sample P&ID is expected to be used as the first extraction test. The agreed extraction target includes equipment such as:

- 1E-2 — CO2 ABSORBER OVERHEAD COOLER
- 1C-3 — CO2 ABSORBER OVERHEAD SEPARATOR
- 1C-4 — AMINE FLASH DRUM

The exact extraction result must always come from the uploaded source PDF, not from hard-coded values.

## Gemini model

The default model is:

```text
gemini-3.5-flash-lite
```

It is selected because the current Gemini documentation describes it as a multimodal model optimized for low-cost/high-throughput tasks and document parsing, with PDF input and structured output support.

To change the model later, edit `GEMINI_MODEL` in `worker/wrangler.jsonc`.

## Security notes

- Never commit `GEMINI_API_KEY`.
- The frontend only knows the Worker URL.
- The Gemini key remains a Cloudflare Worker secret.
- V1 does not permanently store uploaded PDFs.
- The frontend sends the PDF to the Worker as base64 JSON. A 15 MB decoded PDF limit is enforced.
- For larger files or production workloads, the next version should move uploads to R2 and process them without keeping the entire PDF in the request body.

## Next versions

### V2
Improve extraction accuracy and add a source/evidence view.

### V3
Map the extracted records into `template/contoh.xlsx`.

### V4
Add a session:

```text
PDF 001 → Excel
PDF 002 → Excel
PDF 003 → Excel
...
```

### V5
Add editable Excel preview and download.

### V6
Add R2 temporary storage and automatic cleanup.

## Official platform notes

Cloudflare Python Workers supports FastAPI through its ASGI entrypoint and the `python_workers` compatibility flag.

Gemini supports native PDF/document understanding and structured JSON output.

See the current official documentation for Cloudflare Python Workers and Gemini API before deploying, because service limits and model availability can change.
