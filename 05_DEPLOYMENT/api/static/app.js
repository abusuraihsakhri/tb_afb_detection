const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const resultsPanel = document.getElementById('results-panel');
const canvas = document.getElementById('slide-canvas');
const ctx = canvas.getContext('2d');
const hwStatus = document.getElementById('hw-status');
const downloadBtn = document.getElementById('download-btn');

const MAX_FILE_SIZE = 250 * 1024 * 1024;
const VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.jp2', '.j2k', '.jpf', '.jpx'];

let lastAnalysisData = null;
let currentFile = null;
let previewImage = null;

function setStatus(message, visible = true) {
  statusText.textContent = message;
  statusBar.style.display = visible ? 'flex' : 'none';
}

function extensionAllowed(filename) {
  const lower = filename.toLowerCase();
  return VALID_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function resetUI() {
  dropZone.style.display = 'block';
  resultsPanel.style.display = 'none';
  setStatus('', false);
}

function chooseFile() {
  fileInput.click();
}

dropZone.addEventListener('click', chooseFile);
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    chooseFile();
  }
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
});

dropZone.addEventListener('drop', (event) => {
  if (event.dataTransfer?.files?.length) handleFile(event.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (event) => {
  if (event.target.files?.length) handleFile(event.target.files[0]);
});

async function handleFile(file) {
  if (!file) return;
  if (!extensionAllowed(file.name)) {
    alert('Unsupported web-upload format. Use JPG, PNG, TIFF, or an OpenCV-supported JPEG 2000 file.');
    return;
  }
  if (file.size === 0 || file.size > MAX_FILE_SIZE) {
    alert('File must be non-empty and no larger than 250 MB.');
    return;
  }

  currentFile = file;
  lastAnalysisData = null;
  dropZone.style.display = 'none';
  resultsPanel.style.display = 'none';
  setStatus(`Preparing ${file.name}...`);

  try {
    await renderPreview(file);
    await submitForInference(file);
  } catch (error) {
    alert(`Processing error: ${error.message}`);
    resetUI();
  }
}

async function renderPreview(file) {
  const lower = file.name.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.png')) {
    const url = URL.createObjectURL(file);
    try {
      await drawImageUrl(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    return;
  }

  setStatus('Requesting a server-rendered preview...');
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/v1/render_payload', { method: 'POST', body: formData });
  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    throw new Error(detail);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  try {
    await drawImageUrl(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function drawImageUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      previewImage = image;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
      resolve();
    };
    image.onerror = () => reject(new Error('Browser could not render the preview image.'));
    image.src = url;
  });
}

async function safeErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || `Server returned ${response.status}`;
  } catch {
    return `Server returned ${response.status}`;
  }
}

async function submitForInference(file) {
  setStatus('Running screening...');
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/v1/analyze', { method: 'POST', body: formData });
  if (!response.ok) throw new Error(await safeErrorDetail(response));

  const data = await response.json();
  lastAnalysisData = {
    filename: currentFile.name,
    grade: data.grade,
    count: data.candidate_count,
    hardware: data.hardware,
  };
  renderResults(data);
}

function renderResults(data) {
  setStatus('', false);
  resultsPanel.style.display = 'block';

  document.getElementById('smear-grade').textContent = `Smear grade: ${data.grade}`;
  hwStatus.firstElementChild.textContent = data.hardware;
  document.getElementById('analysis-mode').textContent = `Mode: ${data.analysis_mode}`;

  if (previewImage) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(previewImage, 0, 0);
  }

  let afbCount = 0;
  let otherCount = 0;
  const scores = [];

  data.detections.forEach((det) => {
    if (!Array.isArray(det.bbox) || det.bbox.length !== 4) return;
    const [cx, cy, width, height] = det.bbox.map(Number);
    if (![cx, cy, width, height].every(Number.isFinite)) return;

    const classId = Number(det.class_id);
    if (classId >= 0 && classId <= 2) afbCount += 1;
    else otherCount += 1;

    const score = Number(det.confidence);
    if (Number.isFinite(score)) scores.push(score);

    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = classId <= 2 ? '#dc2626' : '#64748b';
    ctx.rect(cx - width / 2, cy - height / 2, width, height);
    ctx.stroke();
  });

  document.getElementById('count-afb').textContent = String(afbCount);
  document.getElementById('count-other').textContent = String(otherCount);
  document.getElementById('score-range').textContent = scores.length
    ? `${(Math.min(...scores) * 100).toFixed(1)}–${(Math.max(...scores) * 100).toFixed(1)}%`
    : '--';
}

downloadBtn.addEventListener('click', async () => {
  if (!lastAnalysisData) return;
  const reviewer = document.getElementById('pathologist-name').value.trim() || 'Not specified';
  setStatus('Generating PDF report...');
  try {
    const response = await fetch('/api/v1/export_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...lastAnalysisData, pathologist_name: reviewer }),
    });
    if (!response.ok) throw new Error(await safeErrorDetail(response));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `TB_AFB_Report_${currentFile.name.replace(/[^a-zA-Z0-9._-]/g, '_')}.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(`Report error: ${error.message}`);
  } finally {
    setStatus('', false);
  }
});
