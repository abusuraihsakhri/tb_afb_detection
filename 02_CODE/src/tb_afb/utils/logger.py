import datetime
import json
from pathlib import Path
from typing import Any


class AuditLogger:
    """Write structured JSON-lines research event logs.

    The logger provides basic event traceability for local research workflows.
    It is not tamper-evident and does not by itself establish regulatory
    compliance.
    """

    def __init__(self, log_dir: Path, user_id: str):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = str(user_id)

    @staticmethod
    def _sanitize(value: Any) -> str:
        return str(value).replace("\n", "\\n").replace("\r", "\\r")

    def _write_log(self, data: dict[str, Any]) -> None:
        safe_data = {str(key): self._sanitize(value) for key, value in data.items()}
        safe_data["user_id"] = self._sanitize(self.user_id)
        safe_data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        log_file = self.log_dir / f"audit_{datetime.datetime.now().strftime('%Y%m')}.log"
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_data) + "\n")

    def log_training_start(self, config_hash: str, data_version: str, git_commit: str) -> None:
        self._write_log(
            {
                "event": "training_start",
                "config": config_hash,
                "data": data_version,
                "commit": git_commit,
            }
        )

    def log_inference(
        self,
        slide_id: str,
        model_version: str,
        result: dict[str, Any],
        processing_time: float,
    ) -> None:
        self._write_log(
            {
                "event": "inference",
                "slide_id": slide_id,
                "model": model_version,
                "time": processing_time,
            }
        )

    def log_data_access(self, data_path: Path, action: str) -> None:
        self._write_log(
            {
                "event": "data_access",
                "path": str(Path(data_path).resolve()),
                "action": action,
            }
        )
