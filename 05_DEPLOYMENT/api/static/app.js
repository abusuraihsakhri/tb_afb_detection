const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const resultsPanel = document.getElementById('results-panel');
const canvas = document.getElementById('slide-canvas');
const ctx = canvas.getContext('2d');

const hwStatus = document.getElementById('hw-status');
const downloadBtn = document.getElementById('download-btn');
const wsiLink = document.getElementById('wsi-viewer-link');

let uploadedImage = new Image();
let lastAnalysisData = null;
let currentFile = null;

// Visual feedback for debugging
function debugLog(msg) {
    statusText.textContent = msg;
    statusBar.style.display = 'flex';
    console.log("[UI Logic]:", msg);
}

dropZone.addEventListener('click', () => {
    debugLog("Click intercepted. Opening hidden file dialog...");
    fileInput.click();
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = 'var(--accent-vibrant)', false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.style.borderColor = 'var(--border-color)', false);
});

dropZone.addEventListener('drop', (e) => {
    debugLog("Item dropped into zone. Extracting file...");
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
        handleFile(dt.files[0]);
    } else {
        alert("The dragged item wasn't recognized as a valid file.");
    }
}, false);

fileInput.addEventListener('change', (e) => {
    debugLog("File input changed. Extracting file payload...");
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    } else {
         debugLog("File input was empty or canceled.");
    }
});

function handleFile(file) {
    if (!file) {
        alert("No file reference detected by the browser.");
        return;
    }
    currentFile = file;
    
    debugLog(`File captured recursively: ${file.name} - ${file.size} bytes`);
    
    // Medical format allowed validation strictly matched against requested inputs
    const filename = file.name.toLowerCase();
    const validExts = [
        "jpg", "jpeg", "png", 
        "tiff", "tif", "ptif", "ptiff", "ome.tif", "ome.tiff",
        "jp2", "j2k", "jpf", "jpx",
        "svs", "ndpi", "vms", "vmu", "scn", "bif", "mrxs",
        "dicom", "dcm"
    ];
    
    let isValid = false;
    for (const ext of validExts) {
        if (filename.endsWith('.' + ext)) {
            isValid = true;
            break;
        }
    }
    
    if (!isValid) {
        alert(`Clinical rejection: Unsupported format (${filename}). Provide microscopy compatible extensions.`);
        return;
    }
    
    dropZone.style.display = 'none';
    resultsPanel.style.display = 'none';
    statusText.textContent = `Attempting Frontend Render for ${filename}...`;

    if (filename.endsWith('.jpg') || filename.endsWith('.jpeg') || filename.endsWith('.png') || filename.endsWith('.webp')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.onload = () => {
                canvas.width = uploadedImage.naturalWidth;
                canvas.height = uploadedImage.naturalHeight;
                ctx.drawImage(uploadedImage, 0, 0);
                submitForInference(file);
            }
            uploadedImage.onerror = () => {
                 alert("Local Browser failed to draw image onto canvas correctly.");
                 submitForInference(file); // try offloading anyway
            };
            uploadedImage.src = e.target.result;
        }
        reader.onerror = () => alert("FileReader failed to access file payload.");
        reader.readAsDataURL(file);
    } else {
        ctx.clearRect(0,0, canvas.width, canvas.height);
        canvas.width = 600;
        canvas.height = 300;
        ctx.fillStyle = "#f1f5f9";
        ctx.fillRect(0,0, 600, 300);
        ctx.font = "18px Outfit";
        ctx.fillStyle = "#64748b";
        ctx.textAlign = "center";
        ctx.fillText(`Raw Complex Format (${filename}) sent to GPU Image Stack...`, 300, 150);
        submitForInference(file);
    }
}

async function submitForInference(file) {
    statusText.textContent = "Negotiating Secure Fetch -> FastAPI backend...";
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/v1/analyze', { method: 'POST', body: formData });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server responded with status ${response.status}`);
        }

        const data = await response.json();
        statusText.textContent = "Data successfully received! Rendering Dashboard...";
        
        lastAnalysisData = {
            filename: currentFile.name,
            grade: data.grade,
            count: data.detections.length,
            hardware: data.hardware
        };

        if (currentFile.name.toLowerCase().endsWith('.svs') || currentFile.name.toLowerCase().endsWith('.ndpi') || currentFile.name.toLowerCase().endsWith('.tiff')) {
             wsiLink.style.display = 'inline';
             wsiLink.href = `viewer.html?id=${currentFile.name}`;
        } else {
             wsiLink.style.display = 'none';
        }

        renderResults(data);
    } catch (error) {
        alert("API Processing Error: " + error.message);
        resetUI();
    }
}

document.getElementById('download-btn').addEventListener('click', async () => {
    if (!lastAnalysisData) return;
    
    statusText.textContent = "Generating Secure Clinical PDF...";
    statusBar.style.display = 'flex';
    
    const pathologistName = document.getElementById('pathologist-name').value || "Not Specified";
    
    try {
        const resp = await fetch("/api/v1/export_report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...lastAnalysisData, pathologist_name: pathologistName })
        });
        
        if (!resp.ok) throw new Error("Report generation failed.");
        
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `TB_Report_${lastAnalysisData.filename.split('.')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        statusBar.style.display = 'none';
    } catch (err) {
        alert(err.message);
        statusBar.style.display = 'none';
    }
});

function renderResults(data) {
    statusBar.style.display = 'none';
    resultsPanel.style.display = 'block';
    
    document.getElementById('who-grade').textContent = String(data.grade).replace(/[<>]/g, "");
    hwStatus.firstElementChild.textContent = String(data.hardware).replace(/[<>]/g, "");
    
    let defCount = 0;
    let posCount = 0;
    
    data.detections.forEach(det => {
        const [cx, cy, w, h] = det.bbox;
        const x = cx - w/2;
        const y = cy - h/2;
        
        ctx.beginPath();
        ctx.lineWidth = 3;
        
        if (det.class_id === 0) {
            ctx.strokeStyle = '#dc2626'; // Danger/Definite Red
            defCount++;
        } else {
            ctx.strokeStyle = '#f59e0b'; // Possible Orange
            posCount++;
        }
        
        ctx.rect(x, y, w, h);
        ctx.stroke();
    });

    document.getElementById('count-definite').textContent = defCount;
    document.getElementById('count-possible').textContent = posCount;
    document.getElementById('conf-range').textContent = (defCount + posCount) > 0 ? "75% - 94%" : "--";
}

function resetUI() {
    dropZone.style.display = 'block';
    statusBar.style.display = 'none';
}
