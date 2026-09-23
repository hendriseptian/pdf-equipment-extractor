/*
  PDF Equipment Extractor V1
  Frontend for GitHub Pages.

  IMPORTANT:
  Set API_BASE_URL to your deployed Cloudflare Worker URL.
  Example:
  const API_BASE_URL = "https://pdf-equipment-extractor.your-subdomain.workers.dev";
*/

const API_BASE_URL = "https://YOUR-WORKER.workers.dev";
const MAX_FILE_MB = 15;

const $ = (selector) => document.querySelector(selector);

const pdfFile = $("#pdfFile");
const dropzone = $("#dropzone");
const fileRow = $("#fileRow");
const fileName = $("#fileName");
const fileSize = $("#fileSize");
const analyzeBtn = $("#analyzeBtn");
const removeFile = $("#removeFile");
const progressCard = $("#progressCard");
const progressTitle = $("#progressTitle");
const progressPercent = $("#progressPercent");
const progressBar = $("#progressBar");
const resultCard = $("#resultCard");
const equipmentCount = $("#equipmentCount");
const resultMessage = $("#resultMessage");
const equipmentList = $("#equipmentList");
const rawJson = $("#rawJson");
const jsonToggle = $("#jsonToggle");
const copyJsonBtn = $("#copyJsonBtn");
const toast = $("#toast");

let selectedFile = null;
let lastResult = null;

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add("hidden"), 3600);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function setProgress(percent, title, activeStep = 0) {
  progressBar.style.width = `${percent}%`;
  progressPercent.textContent = `${percent}%`;
  progressTitle.textContent = title;

  ["p1", "p2", "p3", "p4"].forEach((id, index) => {
    const el = document.getElementById(id);
    el.classList.toggle("active", index === activeStep);
    el.classList.toggle("done", index < activeStep);
  });
}

function resetProgress() {
  progressCard.classList.add("hidden");
  setProgress(0, "Preparing PDF…", 0);
}

function selectFile(file) {
  if (!file) return;

  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Please select a PDF file.", true);
    return;
  }

  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    showToast(`PDF is too large. V1 limit is ${MAX_FILE_MB} MB.`, true);
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileRow.classList.remove("hidden");
  analyzeBtn.disabled = false;
  $("#dropTitle").textContent = "PDF selected";
  $("#dropSubtitle").textContent = "Click here to choose another file";
  resultCard.classList.add("hidden");
  resetProgress();
}

function clearFile() {
  selectedFile = null;
  pdfFile.value = "";
  fileRow.classList.add("hidden");
  analyzeBtn.disabled = true;
  $("#dropTitle").textContent = "Choose a PDF";
  $("#dropSubtitle").textContent = "or drag & drop it here";
  resultCard.classList.add("hidden");
  resetProgress();
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function normalizeValue(value) {
  if (value === null || value === undefined) return "-";
  const s = String(value).trim();
  return (!s || ["null", "none", "n/a", "na", "not found", "unknown"].includes(s.toLowerCase()))
    ? "-"
    : s;
}

function editableFields(equipment, index) {
  const fields = [
    ["tag_no", "Tag No."],
    ["equipment_name", "Equipment Name"],
    ["type", "Type"],
    ["diameter", "Diameter"],
    ["diameter_od_id", "Diameter OD / ID"],
    ["length", "Length"],
    ["length_remarks", "Length Remarks"],
    ["height", "Height"],
    ["insulation", "Insulation"],
    ["insulation_size", "Insulation Size"],
    ["insulation_type", "Insulation Type"],
    ["shell_pressure", "Shell Pressure"],
    ["shell_min_temp", "Shell Min Temp"],
    ["shell_max_temp", "Shell Max Temp"],
    ["tube_pressure", "Tube Pressure"],
    ["tube_min_temp", "Tube Min Temp"],
    ["tube_max_temp", "Tube Max Temp"],
    ["service_type", "Service Type"],
    ["service_description", "Service Description"],
    ["remarks", "Remarks"]
  ];

  return fields.map(([key, label]) => `
    <div class="field">
      <label>${label}</label>
      <input
        data-eq="${index}"
        data-key="${key}"
        value="${escapeHtml(normalizeValue(equipment[key]))}"
        aria-label="${label}"
      >
    </div>
  `).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderResult(data) {
  lastResult = structuredClone(data);
  const equipment = Array.isArray(data.equipment) ? data.equipment : [];

  equipmentCount.textContent = equipment.length;
  resultMessage.textContent = equipment.length
    ? "Review the extracted values. You can edit any field before the Excel stage is added."
    : "No equipment was detected in this PDF.";

  equipmentList.innerHTML = equipment.map((eq, index) => `
    <article class="equipment-card">
      <div class="equipment-card-head">
        <div class="equipment-title">
          <div class="eq-index">${index + 1}</div>
          <div>
            <strong>${escapeHtml(normalizeValue(eq.tag_no))}</strong>
            <span>${escapeHtml(normalizeValue(eq.equipment_name))}</span>
          </div>
        </div>
        <span class="badge">DETECTED</span>
      </div>
      <div class="field-grid">
        ${editableFields(eq, index)}
      </div>
    </article>
  `).join("");

  rawJson.textContent = JSON.stringify(data, null, 2);
  resultCard.classList.remove("hidden");
}

function collectEditedData() {
  if (!lastResult) return null;

  const data = structuredClone(lastResult);
  document.querySelectorAll("[data-eq][data-key]").forEach(input => {
    const index = Number(input.dataset.eq);
    const key = input.dataset.key;
    if (data.equipment?.[index]) {
      data.equipment[index][key] = normalizeValue(input.value);
    }
  });

  lastResult = data;
  rawJson.textContent = JSON.stringify(data, null, 2);
  return data;
}

async function analyze() {
  if (!selectedFile) return;

  if (API_BASE_URL.includes("YOUR-WORKER")) {
    showToast("Set API_BASE_URL in frontend/script.js first.", true);
    return;
  }

  analyzeBtn.disabled = true;
  progressCard.classList.remove("hidden");
  resultCard.classList.add("hidden");

  try {
    setProgress(12, "Reading PDF…", 0);
    const buffer = await selectedFile.arrayBuffer();

    setProgress(28, "Uploading document…", 0);
    const pdfBase64 = arrayBufferToBase64(buffer);

    setProgress(48, "Detecting equipment…", 1);

    const response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: selectedFile.name,
        mime_type: "application/pdf",
        pdf_base64: pdfBase64
      })
    });

    setProgress(72, "Extracting parameters…", 2);

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || payload.error || "Backend request failed.");
    }

    setProgress(92, "Validating…", 3);
    await new Promise(resolve => setTimeout(resolve, 350));

    renderResult(payload);
    setProgress(100, "Analysis complete", 3);
    showToast(`${payload.equipment?.length || 0} equipment detected.`);
  } catch (error) {
    console.error(error);
    progressTitle.textContent = "Analysis failed";
    showToast(error.message || "Unknown error", true);
  } finally {
    analyzeBtn.disabled = !selectedFile;
  }
}

pdfFile.addEventListener("change", e => selectFile(e.target.files[0]));
removeFile.addEventListener("click", clearFile);
analyzeBtn.addEventListener("click", analyze);

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("dragging");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  selectFile(e.dataTransfer.files[0]);
});

jsonToggle.addEventListener("click", () => {
  collectEditedData();
  rawJson.classList.toggle("hidden");
  jsonToggle.innerHTML = rawJson.classList.contains("hidden")
    ? 'Show raw JSON <span>⌄</span>'
    : 'Hide raw JSON <span>⌃</span>';
});

copyJsonBtn.addEventListener("click", async () => {
  const data = collectEditedData();
  if (!data) return;

  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    showToast("JSON copied.");
  } catch {
    showToast("Could not copy JSON.", true);
  }
});
