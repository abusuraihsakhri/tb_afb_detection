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

function setStatus(msg) { statusText.textContent = msg; console.log("[Annotation]:", msg); }

async function refreshKPIs() {
    try {
        const resp = await fetch("/api/v1/stats");
        if(resp.ok) {
            const data = await resp.json();
            document.getElementById("stat-slides").textContent = data.images_annotated;
            document.getElementById("stat-boxes").textContent = data.afb_instances;
            const mBadge = document.getElementById("stat-model");
            if (data.model_deployed) {
                 mBadge.textContent = "HOT-MOUNTED";
                 mBadge.style.color = "var(--success)";
            } else {
                 mBadge.textContent = "Fallback Active";
                 mBadge.style.color = "var(--danger)";
            }
        }
    } catch(err) { console.log("Failed updating dataset schema counts", err); }
}

// Spin up KPIs immediately
refreshKPIs();

dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
});
dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
}, false);
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) loadFile(e.target.files[0]);
});

function loadFile(file) {
    currentFile = file;
    boxes = [];
    
    const filename = file.name.toLowerCase();
    const isBrowserNative = filename.endsWith('.jpg') || filename.endsWith('.jpeg') || filename.endsWith('.png') || filename.endsWith('.webp');

    setStatus(`Processing ${file.name}...`);
    
    if (isBrowserNative) {
        // Direct Client-Side render
        const reader = new FileReader();
        reader.onload = (e) => {
            renderImageToCanvas(e.target.result);
        };
        reader.readAsDataURL(file);
    } else {
        // Server-Side Render for Complex Microscopy formats (TIFF, JP2, DICOM)
        setStatus(`Routing ${filename} to Secure Server Decoder...`);
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/v1/render_payload', { method: 'POST', body: formData })
        .then(async response => {
            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || "Backend CV2 decode failed.");
            }
            return response.blob();
        })
        .then(blob => {
            const objectUrl = URL.createObjectURL(blob);
            renderImageToCanvas(objectUrl);
        })
        .catch(err => {
            alert("Secure Decoder Exception: " + err.message);
            setStatus("Visual Decode Failed.");
            placeholder.style.display = 'block';
        });
    }
}

function renderImageToCanvas(srcUrl) {
    const img = new Image();
    img.onload = () => {
        imgObj = img;
        placeholder.style.display = 'none';
        canvas.style.display = 'block';
        
        // Setup canvas keeping aspect ratio and fitting within 600px height bounding box
        const containerW = canvas.parentElement.clientWidth;
        const containerH = canvas.parentElement.clientHeight;
        
        const wRatio = containerW / img.width;
        const hRatio = containerH / img.height;
        scaleRatio = Math.min(wRatio, hRatio, 1.0);
        
        canvas.width = img.width * scaleRatio;
        canvas.height = img.height * scaleRatio;
        
        redraw();
        setStatus("Ready to annotate " + currentFile.name);
    };
    img.src = srcUrl;
}

function redraw() {
    if (!imgObj) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgObj, 0, 0, canvas.width, canvas.height);
    
    // Draw all saved boxes
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#2563eb'; // Vibrant Blue
    
    boxes.forEach(b => {
        ctx.strokeRect(b.c_x * scaleRatio, b.c_y * scaleRatio, b.c_w * scaleRatio, b.c_h * scaleRatio);
    });
}

// Mouse event tracking to draw rectangles
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currX = e.clientX - rect.left;
    const currY = e.clientY - rect.top;
    
    redraw();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#dc2626'; // Red active box
    ctx.strokeRect(startX, startY, currX - startX, currY - startY);
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    
    const rect = canvas.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    // Calculate raw dimensions
    let rx = Math.min(startX, endX);
    let ry = Math.min(startY, endY);
    let rw = Math.abs(endX - startX);
    let rh = Math.abs(endY - startY);
    
    // Ignore tiny accidental clicks (must be > 5 pixels)
    if (rw > 5 && rh > 5) {
        // Store coordinates un-scaled mapped to YOLO (x_center, y_center, width, height) relative 0.0-1.0
        const true_cx = ((rx + rw/2) / scaleRatio) / imgObj.naturalWidth;
        const true_cy = ((ry + rh/2) / scaleRatio) / imgObj.naturalHeight;
        const true_w = (rw / scaleRatio) / imgObj.naturalWidth;
        const true_h = (rh / scaleRatio) / imgObj.naturalHeight;
        
        boxes.push({
            c_x: rx / scaleRatio, c_y: ry / scaleRatio, c_w: rw / scaleRatio, c_h: rh / scaleRatio,
            yolo_x: true_cx, yolo_y: true_cy, yolo_w: true_w, yolo_h: true_h,
            label: 0
        });
        setStatus(`Added Box ${boxes.length}`);
    }
    redraw();
});

document.getElementById('clear-btn').addEventListener('click', () => { boxes = []; redraw(); setStatus("Cleared boxes."); });

document.getElementById('submit-btn').addEventListener('click', async () => {
    if (!currentFile || boxes.length === 0) {
        alert("Clinical Form Error: Cannot dispatch empty JSON array. Please upload and box an image.");
        return;
    }
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Format boxes for Python Pydantic Schema
    const uploadBoxes = boxes.map(b => ({
       x: b.yolo_x, y: b.yolo_y, width: b.yolo_w, height: b.yolo_h, label: b.label
    }));
    formData.append('boxes', JSON.stringify(uploadBoxes));
    
    setStatus("Encrypting boundaries and uploading to GPU Volume...");
    
    try {
        const resp = await fetch("/api/v1/save_annotation", { method: "POST", body: formData });
        const json = await resp.json();
        if(!resp.ok) throw new Error(json.detail);
        
        setStatus(json.message);
        boxes = [];
        redraw();
        refreshKPIs(); // Refresh visual count safely
    } catch (err) {
        alert("API Save Failure: " + err.message);
        setStatus("Save failed.");
    }
});

document.getElementById('train-btn').addEventListener('click', async () => {
    setStatus("Firing CUDA Model Overhaul Request...");
    try {
        const resp = await fetch("/api/v1/trigger_training", { method: "POST" });
        const json = await resp.json();
        if(!resp.ok) throw new Error(json.detail);
        setStatus(json.message);
    } catch (err) {
        alert("Training Thread Isolation Error: " + err.message);
        setStatus("Training denied.");
    }
});
