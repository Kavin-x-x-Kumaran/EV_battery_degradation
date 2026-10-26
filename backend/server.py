from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from agents import Orchestrator


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
orchestrator = Orchestrator(ROOT)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json({"ready": orchestrator.model_path.exists(), "title": PROJECT_TITLE})
            return
        if path == "/api/train":
            bundle = orchestrator.train_or_load()
            self._json({"ready": True, "metadata": bundle.metadata})
            return
        if path == "/api/telemetry":
            self._json(orchestrator.telemetry())
            return
        if path.startswith("/api/"):
            self._json({"error": "not found"}, status=404)
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self._json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self._json(orchestrator.predict(payload))

    def _static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"
        target = (FRONTEND / path.lstrip("/")).resolve()
        if not str(target).startswith(str(FRONTEND.resolve())) or not target.exists():
            self.send_error(404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("content-type", mimetypes.guess_type(target)[0] or "application/octet-stream")
        self.send_header("content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


PROJECT_TITLE = (
    "Multi-Agent Explainable AI Framework for Early Life Prediction of Long-Term EV "
    "Battery Degradation Using Sparse Charging Cycles, Features and Digital Twin Feedback"
)


def main() -> None:
    host, port = "127.0.0.1", 8000
    print(f"Starting EV battery digital twin demo at http://{host}:{port}")
    print("First /api/train call builds the model cache from the Parquet zip.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
