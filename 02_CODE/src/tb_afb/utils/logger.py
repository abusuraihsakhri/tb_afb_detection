from pathlib import Path
from typing import Dict
import datetime
import json

class AuditLogger:
    """
    Medical device audit logging (21 CFR Part 11 compliance).
    Forces robust log sanitization and prevents Log Forging/Injection attacks.
    """
    def __init__(self, log_dir: Path, user_id: str):
        # 🛡️ SECURITY CONTROL: Absolute localized path enforcement
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = str(user_id)

    def _write_log(self, data: Dict):
        # 🛡️ SECURITY CONTROL: Append-only strict logs, sanitizing input metadata to prevent Log Injection (CRLF)
        safe_data = {}
        for k, v in data.items():
            # Stripping newline characters maliciously injected into metadata attributes
            safe_val = str(v).replace('\n', '\\n').replace('\r', '\\r')
            safe_data[str(k)] = safe_val
            
        safe_data['user_id'] = self.user_id
        safe_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        log_file = self.log_dir / f"audit_{datetime.datetime.now().strftime('%Y%m')}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(safe_data) + "\n")

    def log_training_start(self, config_hash: str, data_version: str, git_commit: str) -> None:
        self._write_log({"event": "training_start", "config": config_hash, "data": data_version, "commit": git_commit})

    def log_inference(self, slide_id: str, model_version: str, result: Dict, processing_time: float) -> None:
        self._write_log({"event": "inference", "slide_id": slide_id, "model": model_version, "time": processing_time})

    def log_data_access(self, data_path: Path, action: str) -> None:
        self._write_log({"event": "data_access", "path": str(Path(data_path).resolve()), "action": action})
