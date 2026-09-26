# Security Policy

## Reporting a vulnerability

Please use GitHub Private Vulnerability Reporting when available. If private reporting is unavailable, contact the repository maintainer directly rather than publishing exploit details in a public issue.

Include the affected file or endpoint, reproduction steps, expected impact, and any relevant environment details.

## Supported code

Security fixes are applied to the current `main` branch. There are no separately maintained release branches at present.

## Threat model and deployment notes

- The default local launch instructions bind FastAPI to `127.0.0.1`.
- Docker Compose publishes port 8001 on loopback only.
- Uploads are bounded by actual bytes read and must decode as supported raster images before analysis or annotation storage.
- Annotation coordinates are validated as normalized in-image boxes before they are written.
- Training configuration paths and CLI data paths are confined to their intended repository directories.
- Trained checkpoints are discovered only under local model-output directories.
- Checkpoints are trusted inputs. Do not load arbitrary `.pt` files obtained from untrusted sources.
- Patient-derived data, model outputs, and logs are excluded from Git by default, but users remain responsible for de-identification and access control on the host system.

## Non-goals

This project is research software, not a certified clinical or security-hardened medical device. The repository does not claim regulatory compliance, diagnostic validation, or suitability for direct Internet exposure.
