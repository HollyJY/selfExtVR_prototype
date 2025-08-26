import json, time
from typing import Any, Dict, List

class Timeline:
    def __init__(self, path: str):
        self.path = path
        self.buf: List[Dict[str, Any]] = []

    def add(self, event: str, **payload):
        rec = {"ts": time.time(), "event": event, "payload": payload}
        self.buf.append(rec)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def snapshot(self):
        return self.buf
