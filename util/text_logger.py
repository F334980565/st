import json
import os
from datetime import datetime

class TextLogger:
    def __init__(self, log_dir, filename="run_log.txt"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.log(
            f"=== logger started at {datetime.now().isoformat(timespec='seconds')} ==="
        )

    def _format(self, message, step=None, prefix=None):
        if isinstance(message, (dict, list, tuple)):
            message = json.dumps(
                message, ensure_ascii=False, sort_keys=True, default=str
            )
        else:
            message = str(message)
        parts = [f"[{datetime.now().isoformat(timespec='seconds')}]"]
        if step is not None:
            parts.append(f"step={step}")
        if prefix:
            parts.append(prefix)
        parts.append(message)
        return " ".join(parts)

    def log(self, message, step=None, prefix=None):
        line = self._format(message, step=step, prefix=prefix)
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
