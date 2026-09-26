const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const resultsPanel = document.getElementById('results-panel');
const canvas = document.getElementById('slide-canvas');
const ctx = canvas.getContext('2d');
const hwStatus = document.getElementById('hw-status');
const wsiLink = document.getElementById('wsi-viewer-link');

let uploadedImage = new Image();
let lastAnalysisData = null;
let currentFile = null;

function setStatus(message) {
    statusText.textContent = message;
    statusBar.style.display = 'flex';
}

function preventDefaults(event) {
    event.preventDefault();
    event.stopPropagation();
}

dropZone.addEventListener('click', () => fileInput.click());
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, preventDefaults));
['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, () => {
    dropZone.style.borderColor = 'var(--accent-vibrant)';
}));
['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, () => {
    dropZone.style.borderColor = 'var(--border-color)';
}));

dropZone.addEventListener('drop', event => {
    if (event.dataTransfer?.files?.length) handleFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', event => {
    if (event.target.files?.length) handleFile(event.target.files[0]);
});

function handleFile(file) {
    const filename = file.name.toLowerCase();
    const validExtensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.jp2', '.j2k', '.jpf', '.jpx'];
    if (!validExtensions.some(ext => filename.endsWith(ext))) {
        alert('Unsupported web-UI format. Use the CLI WSI workflow for SVS/NDPI and other OpenSlide formats.');
        return;
    }

    currentFile = file;
    dropZone.style.display = 'none';
    resultsPanel.style.display = 'none';
    setStatus(`Preparing ${file.name}...`);

    if (['.jpg', '.jpeg', '.png'].some(ext => filename.endsWith(ext))) {
        const reader = new FileReader();
        reader.onload = event => {
            uploadedImage.onload = () => {
                canvas.width = uploadedImage.naturalWidth;
                canvas.height = uploadedImage.naturalHeight;
                ctx.drawImage(uploadedImage, 0, 0);
                submitForInference(file);
            };
            uploadedImage.onerror = () => submitForInference(file);
            uploadedImage.src = event.target.result;
        };
        reader.onerror = () => {
            alert('The browser could not read this file.');
            resetUI();
        };
        reader.readAsDataURL(file);
    } else {
        canvas.width = 600;
        canvas.height = 300;
        ctx.fillStyle = '#f1f5f9';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '18px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Preview will be decoded by the local server.', 300, 150);
        submitForInference(file);
    }
}

async function submitForInference(file) {
    setStatus('Running local research inference...');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/v1/analyze', { method: 'POST', body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);

        lastAnalysisData = {
            filename: currentFile.name,
            grade: payload.grade,
            count: payload.detections.length,
            hardware: payload.hardware,
        };
        wsiLink.style.display = payload.wsi_id ? 'inline' : 'none';
        if (payload.wsi_id) wsiLink.href = `viewer.html?id=${encodeURIComponent(payload.wsi_id)}`;
        renderResults(payload);
    } catch (error) {
        alert(`Processing error: ${error.message}`);
        resetUI();
    }
}

document.getElementById('download-btn').addEventListener('click', async () => {
    if (!lastAnalysisData) return;
    setStatus('Generating local PDF report...');
    const pathologistName = document.getElementById('pathologist-name').value || 'Not specified';

    try {
        const response = await fetch('/api/v1/export_report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...lastAnalysisData, pathologist_name: pathologistName }),
        });
        if (!response.ok) throw new Error('Report generation failed.');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `TB_AFB_Research_${lastAnalysisData.filename.split('.')[0]}.pdf`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
    } catch (error) {
        alert(error.message);
    } finally {
        statusBar.style.display = 'none';
    }
});

function renderResults(data) {
    statusBar.style.display = 'none';
    resultsPanel.style.display = 'block';
    document.getElementById('who-grade').textContent = String(data.grade);
    hwStatus.firstElementChild.textContent = String(data.hardware);

    let definiteCount = 0;
    let possibleCount = 0;
    const confidences = [];

    data.detections.forEach(det => {
        const [cx, cy, width, height] = det.bbox;
        const x = cx - width / 2;
        const y = cy - height / 2;
        const confidence = Number(det.confidence);
        if (Number.isFinite(confidence)) confidences.push(confidence);

        ctx.beginPath();
        ctx.lineWidth = 3;
        if (det.class_id === 0) {
            ctx.strokeStyle = '#dc2626';
            definiteCount += 1;
        } else {
            ctx.strokeStyle = '#f59e0b';
            possibleCount += 1;
        }
        ctx.rect(x, y, width, height);
        ctx.stroke();
    });

    document.getElementById('count-definite').textContent = definiteCount;
    document.getElementById('count-possible').textContent = possibleCount;
    const confidenceText = confidences.length
        ? `${(Math.min(...confidences) * 100).toFixed(1)}% - ${(Math.max(...confidences) * 100).toFixed(1)}%`
        : '--';
    document.getElementById('conf-range').textContent = confidenceText;
}

function resetUI() {
    dropZone.style.display = 'block';
    statusBar.style.display = 'none';
    resultsPanel.style.display = 'none';
}
