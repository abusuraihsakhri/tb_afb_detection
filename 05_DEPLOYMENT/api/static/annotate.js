const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const canvas = document.getElementById('annotate-canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('canvas-placeholder');
const statusText = document.getElementById('status-text');
const trainButton = document.getElementById('train-btn');

const MAX_FILE_SIZE = 250 * 1024 * 1024;
const VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.jp2', '.j2k', '.jpf', '.jpx'];

let isDrawing = false;
let startX = 0;
let startY = 0;
let boxes = [];
let imgObj = null;
let currentFile = null;
let scaleRatio = 1.0;
let csrfToken = null;

function setStatus(message) {
  statusText.textContent = message;
}

function extensionAllowed(filename) {
  const lower = filename.toLowerCase();
  return VALID_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

async function safeErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || `Server returned ${response.status}`;
  } catch {
    return `Server returned ${response.status}`;
  }
}

async function ensureCsrfToken() {
  if (csrfToken) return csrfToken;
  const response = await fetch('/api/v1/session', { cache: 'no-store' });
  if (!response.ok) throw new Error(await safeErrorDetail(response));
  const data = await response.json();
  if (!data.csrf_token) throw new Error('Server did not provide a request token.');
  csrfToken = data.csrf_token;
  return csrfToken;
}

async function refreshKPIs() {
  try {
    const response = await fetch('/api/v1/stats');
    if (!response.ok) return;
    const data = await response.json();
    document.getElementById('stat-slides').textContent = data.images_annotated;
    document.getElementById('stat-boxes').textContent = data.afb_instances;
    const modelBadge = document.getElementById('stat-model');
    modelBadge.textContent = data.model_deployed ? 'Available' : 'Not deployed';
    trainButton.disabled = !data.training_trigger_enabled;
    trainButton.title = data.training_trigger_enabled
      ? 'Start a local training process'
      : 'Disabled by default; use the trusted local annotation launcher to enable it';
  } catch {
    setStatus('Could not read dataset status.');
  }
}

refreshKPIs();

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
});

dropZone.addEventListener('drop', (event) => {
  if (event.dataTransfer?.files?.length) loadFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (event) => {
  if (event.target.files?.length) loadFile(event.target.files[0]);
});

async function loadFile(file) {
  if (!extensionAllowed(file.name)) {
    alert('Unsupported annotation format.');
    return;
  }
  if (file.size === 0 || file.size > MAX_FILE_SIZE) {
    alert('File must be non-empty and no larger than 250 MB.');
    return;
  }

  currentFile = file;
  boxes = [];
  setStatus(`Preparing ${file.name}...`);

  const lower = file.name.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.png')) {
    const url = URL.createObjectURL(file);
    try {
      await renderImageToCanvas(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    return;
  }

  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await fetch('/api/v1/render_payload', { method: 'POST', body: formData });
    if (!response.ok) throw new Error(await safeErrorDetail(response));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    try {
      await renderImageToCanvas(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  } catch (error) {
    alert(`Preview error: ${error.message}`);
    setStatus('Preview failed.');
  }
}

function renderImageToCanvas(srcUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      imgObj = image;
      placeholder.style.display = 'none';
      canvas.style.display = 'block';
      const containerW = canvas.parentElement.clientWidth;
      const containerH = canvas.parentElement.clientHeight;
      scaleRatio = Math.min(containerW / image.width, containerH / image.height, 1.0);
      canvas.width = Math.max(1, Math.round(image.width * scaleRatio));
      canvas.height = Math.max(1, Math.round(image.height * scaleRatio));
      redraw();
      setStatus(`Ready to annotate ${currentFile.name}.`);
      resolve();
    };
    image.onerror = () => reject(new Error('Could not render image.'));
    image.src = srcUrl;
  });
}

function redraw() {
  if (!imgObj) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#2563eb';
  boxes.forEach((box) => {
    ctx.strokeRect(box.rawX * scaleRatio, box.rawY * scaleRatio, box.rawW * scaleRatio, box.rawH * scaleRatio);
  });
}

canvas.addEventListener('pointerdown', (event) => {
  if (!imgObj) return;
  isDrawing = true;
  canvas.setPointerCapture(event.pointerId);
  const rect = canvas.getBoundingClientRect();
  startX = event.clientX - rect.left;
  startY = event.clientY - rect.top;
});

canvas.addEventListener('pointermove', (event) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const currentX = Math.min(Math.max(event.clientX - rect.left, 0), canvas.width);
  const currentY = Math.min(Math.max(event.clientY - rect.top, 0), canvas.height);
  redraw();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#dc2626';
  ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
});

canvas.addEventListener('pointerup', (event) => {
  if (!isDrawing) return;
  isDrawing = false;
  const rect = canvas.getBoundingClientRect();
  const endX = Math.min(Math.max(event.clientX - rect.left, 0), canvas.width);
  const endY = Math.min(Math.max(event.clientY - rect.top, 0), canvas.height);

  const x = Math.min(startX, endX);
  const y = Math.min(startY, endY);
  const width = Math.abs(endX - startX);
  const height = Math.abs(endY - startY);

  if (width > 5 && height > 5) {
    const rawX = x / scaleRatio;
    const rawY = y / scaleRatio;
    const rawW = width / scaleRatio;
    const rawH = height / scaleRatio;
    boxes.push({
      rawX,
      rawY,
      rawW,
      rawH,
      x: (rawX + rawW / 2) / imgObj.naturalWidth,
      y: (rawY + rawH / 2) / imgObj.naturalHeight,
      width: rawW / imgObj.naturalWidth,
      height: rawH / imgObj.naturalHeight,
      label: 0,
    });
    setStatus(`Boxes: ${boxes.length}`);
  }
  redraw();
});

document.getElementById('clear-btn').addEventListener('click', () => {
  boxes = [];
  redraw();
  setStatus('Boxes cleared.');
});

document.getElementById('submit-btn').addEventListener('click', async () => {
  if (!currentFile || boxes.length === 0) {
    alert('Choose an image and draw at least one box first.');
    return;
  }
  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('boxes', JSON.stringify(boxes.map(({ x, y, width, height, label }) => ({ x, y, width, height, label }))));
  setStatus('Saving annotation...');
  try {
    const token = await ensureCsrfToken();
    const response = await fetch('/api/v1/save_annotation', {
      method: 'POST',
      headers: { 'X-CSRF-Token': token },
      body: formData,
    });
    if (!response.ok) throw new Error(await safeErrorDetail(response));
    const data = await response.json();
    boxes = [];
    redraw();
    setStatus(data.message);
    await refreshKPIs();
  } catch (error) {
    alert(`Save error: ${error.message}`);
    setStatus('Save failed.');
  }
});

trainButton.addEventListener('click', async () => {
  setStatus('Starting training...');
  try {
    const token = await ensureCsrfToken();
    const response = await fetch('/api/v1/trigger_training', {
      method: 'POST',
      headers: { 'X-CSRF-Token': token },
    });
    if (!response.ok) throw new Error(await safeErrorDetail(response));
    const data = await response.json();
    setStatus(data.message);
  } catch (error) {
    alert(`Training error: ${error.message}`);
    setStatus('Training not started.');
  }
});
