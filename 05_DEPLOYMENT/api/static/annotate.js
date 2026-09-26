const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const canvas = document.getElementById('annotate-canvas');
const ctx = canvas.getContext('2d');
const placeholder = document.getElementById('canvas-placeholder');
const statusText = document.getElementById('status-text');

let isDrawing = false;
let startX = 0;
let startY = 0;
let boxes = [];
let imgObj = null;
let currentFile = null;
let scaleRatio = 1.0;

function setStatus(message) { statusText.textContent = message; }

async function refreshKPIs() {
    try {
        const response = await fetch('/api/v1/stats');
        if (!response.ok) return;
        const data = await response.json();
        document.getElementById('stat-slides').textContent = data.images_annotated;
        document.getElementById('stat-boxes').textContent = data.afb_instances;
        const badge = document.getElementById('stat-model');
        badge.textContent = data.model_deployed ? 'Checkpoint available' : 'No trained checkpoint';
        badge.style.color = data.model_deployed ? 'var(--success)' : 'var(--danger)';
    } catch (_) {
        setStatus('Could not read dataset statistics.');
    }
}

refreshKPIs();
dropZone.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, event => { event.preventDefault(); event.stopPropagation(); });
});
dropZone.addEventListener('drop', event => {
    if (event.dataTransfer.files.length) loadFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', event => {
    if (event.target.files.length) loadFile(event.target.files[0]);
});

function loadFile(file) {
    const filename = file.name.toLowerCase();
    const allowed = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.jp2', '.j2k', '.jpf', '.jpx'];
    if (!allowed.some(ext => filename.endsWith(ext))) {
        alert('Unsupported annotation format.');
        return;
    }

    currentFile = file;
    boxes = [];
    setStatus(`Preparing ${file.name}...`);

    if (['.jpg', '.jpeg', '.png'].some(ext => filename.endsWith(ext))) {
        const reader = new FileReader();
        reader.onload = event => renderImageToCanvas(event.target.result);
        reader.readAsDataURL(file);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    fetch('/api/v1/render_payload', { method: 'POST', body: formData })
        .then(async response => {
            if (!response.ok) throw new Error((await response.json()).detail || 'Decode failed.');
            return response.blob();
        })
        .then(blob => renderImageToCanvas(URL.createObjectURL(blob)))
        .catch(error => {
            alert(error.message);
            setStatus('Image decode failed.');
        });
}

function renderImageToCanvas(srcUrl) {
    const img = new Image();
    img.onload = () => {
        imgObj = img;
        placeholder.style.display = 'none';
        canvas.style.display = 'block';
        const containerW = canvas.parentElement.clientWidth;
        const containerH = canvas.parentElement.clientHeight;
        scaleRatio = Math.min(containerW / img.width, containerH / img.height, 1.0);
        canvas.width = img.width * scaleRatio;
        canvas.height = img.height * scaleRatio;
        redraw();
        setStatus(`Ready to annotate ${currentFile.name}`);
        if (srcUrl.startsWith('blob:')) URL.revokeObjectURL(srcUrl);
    };
    img.onerror = () => setStatus('Image preview failed.');
    img.src = srcUrl;
}

function redraw() {
    if (!imgObj) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#2563eb';
    boxes.forEach(box => ctx.strokeRect(box.c_x * scaleRatio, box.c_y * scaleRatio, box.c_w * scaleRatio, box.c_h * scaleRatio));
}

canvas.addEventListener('mousedown', event => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = event.clientX - rect.left;
    startY = event.clientY - rect.top;
});
canvas.addEventListener('mousemove', event => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currX = event.clientX - rect.left;
    const currY = event.clientY - rect.top;
    redraw();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#dc2626';
    ctx.strokeRect(startX, startY, currX - startX, currY - startY);
});
canvas.addEventListener('mouseup', event => {
    if (!isDrawing) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const endX = event.clientX - rect.left;
    const endY = event.clientY - rect.top;
    const rx = Math.min(startX, endX);
    const ry = Math.min(startY, endY);
    const rw = Math.abs(endX - startX);
    const rh = Math.abs(endY - startY);

    if (rw > 5 && rh > 5) {
        const x = ((rx + rw / 2) / scaleRatio) / imgObj.naturalWidth;
        const y = ((ry + rh / 2) / scaleRatio) / imgObj.naturalHeight;
        const width = (rw / scaleRatio) / imgObj.naturalWidth;
        const height = (rh / scaleRatio) / imgObj.naturalHeight;
        boxes.push({ c_x: rx / scaleRatio, c_y: ry / scaleRatio, c_w: rw / scaleRatio, c_h: rh / scaleRatio, x, y, width, height, label: 0 });
        setStatus(`Added box ${boxes.length}`);
    }
    redraw();
});

document.getElementById('clear-btn').addEventListener('click', () => { boxes = []; redraw(); setStatus('Cleared boxes.'); });
document.getElementById('submit-btn').addEventListener('click', async () => {
    if (!currentFile || boxes.length === 0) {
        alert('Upload an image and draw at least one box.');
        return;
    }
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('boxes', JSON.stringify(boxes.map(({ x, y, width, height, label }) => ({ x, y, width, height, label }))));
    setStatus('Saving normalized annotation...');
    try {
        const response = await fetch('/api/v1/save_annotation', { method: 'POST', body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Save failed.');
        boxes = [];
        redraw();
        setStatus(payload.message);
        refreshKPIs();
    } catch (error) {
        alert(error.message);
        setStatus('Save failed.');
    }
});

document.getElementById('train-btn').addEventListener('click', async () => {
    setStatus('Requesting local training process...');
    try {
        const response = await fetch('/api/v1/trigger_training', { method: 'POST' });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Training could not start.');
        setStatus(payload.message);
        refreshKPIs();
    } catch (error) {
        alert(error.message);
        setStatus('Training request failed.');
    }
});
