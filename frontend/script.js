const API_BASE = "https://pdf-equipment-extractor.side-gs78.workers.dev";

const pdfInput = document.getElementById("pdfInput");
const dropzone = document.getElementById("dropzone");
const fileTitle = document.getElementById("fileTitle");
const fileMeta = document.getElementById("fileMeta");
const analyzeBtn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const spinner = document.getElementById("spinner");
const clearBtn = document.getElementById("clearBtn");

const errorBox = document.getElementById("errorBox");
const successBox = document.getElementById("successBox");

const resultSection = document.getElementById("resultSection");
const resultBody = document.getElementById("resultBody");
const equipmentCount = document.getElementById("equipmentCount");
const rawJson = document.getElementById("rawJson");

const apiStatus = document.getElementById("apiStatus");
const statusDot = document.querySelector(".status-dot");

let selectedFile = null;

const MAX_FILE_SIZE = 15 * 1024 * 1024;

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  successBox.classList.add("hidden");
}

function showSuccess(message) {
  successBox.textContent = message;
  successBox.classList.remove("hidden");
  errorBox.classList.add("hidden");
}

function clearMessages() {
  errorBox.classList.add("hidden");
  successBox.classList.add("hidden");
}

function escapeHtml(value) {
  return String(value ?? "-")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setFile(file) {
  clearMessages();

  if (!file) return;

  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    selectedFile = null;
    analyzeBtn.disabled = true;
    showError("Please select a PDF file.");
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    selectedFile = null;
    analyzeBtn.disabled = true;
    showError("PDF is too large. Maximum size is 15 MB.");
    return;
  }

  selectedFile = file;
  fileTitle.textContent = file.name;
  fileMeta.textContent = `${formatSize(file.size)} • PDF ready for analysis`;
  analyzeBtn.disabled = false;
}

pdfInput.addEventListener("change", () => {
  setFile(pdfInput.files[0]);
});

["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  setFile(file);
});

function readAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const result = reader.result;

      if (typeof result !== "string") {
        reject(new Error("Could not read PDF."));
        return;
      }

      const commaIndex = result.indexOf(",");

      if (commaIndex === -1) {
        reject(new Error("Invalid PDF data."));
        return;
      }

      resolve(result.substring(commaIndex + 1));
    };

    reader.onerror = () => reject(new Error("Could not read PDF."));
    reader.readAsDataURL(file);
  });
}

function setLoading(loading) {
  analyzeBtn.disabled = loading || !selectedFile;
  btnText.textContent = loading ? "Analyzing..." : "Analyze PDF";
  spinner.classList.toggle("hidden", !loading);
}

function renderResults(data) {
  const equipment = Array.isArray(data.equipment) ? data.equipment : [];

  equipmentCount.textContent = equipment.length;
  resultBody.innerHTML = "";

  equipment.forEach((item, index) => {
    const row = document.createElement("tr");

    const shellTemp = [
      item.shell_min_temp && item.shell_min_temp !== "-" ? `Min: ${item.shell_min_temp}` : null,
      item.shell_max_temp && item.shell_max_temp !== "-" ? `Max: ${item.shell_max_temp}` : null
    ].filter(Boolean).join(" / ") || "-";

    const tubeTemp = [
      item.tube_min_temp && item.tube_min_temp !== "-" ? `Min: ${item.tube_min_temp}` : null,
      item.tube_max_temp && item.tube_max_temp !== "-" ? `Max: ${item.tube_max_temp}` : null
    ].filter(Boolean).join(" / ") || "-";

    const cells = [
      index + 1,
      item.tag_no,
      item.equipment_name,
      item.type,
      item.diameter,
      item.length,
      item.height,
      item.insulation,
      item.shell_pressure,
      shellTemp,
      item.tube_pressure,
      tubeTemp,
      item.service_description,
      item.remarks
    ];

    cells.forEach(value => {
      const td = document.createElement("td");
      td.textContent = value ?? "-";
      row.appendChild(td);
    });

    resultBody.appendChild(row);
  });

  rawJson.textContent = JSON.stringify(data, null, 2);
  resultSection.classList.remove("hidden");
}

async function analyzePdf() {
  if (!selectedFile) return;

  clearMessages();
  resultSection.classList.add("hidden");
  setLoading(true);

  try {
    const pdfBase64 = await readAsBase64(selectedFile);

    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        filename: selectedFile.name,
        mime_type: "application/pdf",
        pdf_base64: pdfBase64
      })
    });

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error(`Server returned HTTP ${response.status} with an invalid JSON response.`);
    }

    if (!response.ok) {
      throw new Error(data?.detail || `Server returned HTTP ${response.status}.`);
    }

    renderResults(data);
    showSuccess(`Analysis completed. ${data.equipment_count ?? 0} equipment detected.`);
  } catch (error) {
    console.error(error);
    showError(error.message || "Analysis failed.");
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener("click", analyzePdf);

clearBtn.addEventListener("click", () => {
  selectedFile = null;
  pdfInput.value = "";
  fileTitle.textContent = "Choose a PDF file";
  fileMeta.textContent = "or drag & drop it here";
  analyzeBtn.disabled = true;
  resultSection.classList.add("hidden");
  resultBody.innerHTML = "";
  rawJson.textContent = "";
  clearMessages();
});

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: "GET",
      cache: "no-store"
    });

    if (!response.ok) throw new Error();

    const data = await response.json();

    apiStatus.textContent = `API Online • ${data.model || "Gemini"}`;
    statusDot.classList.add("online");
  } catch {
    apiStatus.textContent = "API Offline";
    statusDot.classList.add("offline");
  }
}

checkApi();
