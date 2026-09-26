# Security Policy

## Reporting a vulnerability

Please use GitHub Private Vulnerability Reporting for security issues when it is available for this repository. Do not include patient data, credentials, or other sensitive material in a public issue.

A useful report includes the affected component, reproduction steps, expected impact, and the dependency/runtime versions involved.

## Security model

This project is designed primarily for trusted local research use. The supplied launchers bind the FastAPI service to `127.0.0.1`, and Docker Compose also binds port 8001 to localhost.

The API does not implement user authentication or authorization. CORS restrictions are a browser control and are **not** an authentication mechanism. Do not expose the application directly to an untrusted network. For shared deployment, place it behind an authenticated reverse proxy and configure transport security, request limits, logging, and host-level access controls.

## Implemented controls

- Path resolution uses `Path.relative_to`-based containment checks for configured data roots.
- Raw HTTP request bodies, streamed raster uploads, and decoded pixel counts are bounded.
- Trusted-host validation restricts the supplied local service to `localhost`/`127.0.0.1` host headers, reducing DNS-rebinding exposure.
- Accepted web-upload extensions are restricted and the image must decode successfully.
- Annotation bounding boxes are validated to normalized image bounds.
- Annotation images are decoded and re-encoded before entering the training dataset.
- State-changing annotation and training requests require an ephemeral same-origin request token, reducing cross-origin form/CSRF abuse in the local browser workflow.
- The HTTP training trigger is disabled unless `TB_AFB_ENABLE_TRAINING_TRIGGER=1` is explicitly set.
- Only one training process can be launched at a time through the API.
- WSI region reads and the in-memory WSI handle cache are bounded.
- Clinical/research data, trained models, run outputs, and logs are ignored by Git by default.
- The multipart parser is pinned to a release containing the currently known 2026 parser fixes.

## Model checkpoint safety

Only load `.pt` checkpoints from trusted sources. PyTorch/Ultralytics checkpoints are not treated as untrusted data by this project, and the repository does not claim to sandbox model deserialization.

The committed `yolov8n.pt` file is a generic initialization checkpoint, not an AFB-validated diagnostic model.

## Data privacy

The application processes files locally by default, but local processing alone is not a privacy guarantee. Git ignore rules do not de-identify images, remove embedded metadata, encrypt data at rest, or enforce retention policy.

Use only data that your institution permits for the intended environment. Review slide metadata and logs for PHI/PII before sharing artifacts.

## Third-party components

The optional WSI viewer page loads OpenSeadragon 4.1.0 from cdnjs. Slide pixels remain served by the local API, but opening that page makes a request to the CDN for the viewer library/assets. Environments requiring fully offline operation should vendor or block that optional page.

Dependency vulnerabilities relevant to this application may be reported here even when the underlying fix belongs upstream.
