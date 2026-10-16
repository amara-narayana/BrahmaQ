"""Lightweight HTTP API & Static File Server for BrahmaQ (Q-TMCE) Frontend UI.

Uses Python standard library only (no extra pip dependencies required).
Serves the rich interactive frontend UI and provides REST API endpoints to
execute the engine, query artifacts, and run cryptographic verification.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UI_DIR = PROJECT_ROOT / "ui"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_ARTIFACT = ARTIFACTS_DIR / "qtmce_result.json"
DEFAULT_KEYSTORE = ARTIFACTS_DIR / "zkp_keystore.json"


class BrahmaQRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing API endpoints and static UI serving."""

    server_version = "BrahmaQ-Server/1.0"

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"
        )

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: Any) -> None:
        """Helper to send a JSON response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status_code: int, message: str, details: Optional[Any] = None) -> None:
        """Send formatted JSON error response."""
        self._send_json(status_code, {"error": message, "details": details, "timestamp": time.time()})

    def do_GET(self) -> None:
        """Route GET requests to API or static UI assets."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # API: Status
        if path == "/api/status":
            try:
                import pyscf  # type: ignore # noqa: F401
                has_pyscf = True
            except ImportError:
                has_pyscf = False

            self._send_json(200, {
                "name": "BrahmaQ (Q-TMCE)",
                "status": "online",
                "version": "1.0.0",
                "python": sys.version,
                "pyscf_available": has_pyscf,
                "artifact_exists": DEFAULT_ARTIFACT.exists(),
                "keystore_exists": DEFAULT_KEYSTORE.exists(),
            })
            return

        # API: Latest Artifact
        if path == "/api/artifact":
            if DEFAULT_ARTIFACT.exists():
                try:
                    with DEFAULT_ARTIFACT.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(200, data)
                    return
                except Exception as exc:
                    self._send_error_json(500, f"Error reading artifact: {exc}")
                    return
            else:
                self._send_error_json(404, "Artifact not yet generated. Run the engine first.")
                return

        # API: Keystore
        if path == "/api/keystore":
            if DEFAULT_KEYSTORE.exists():
                try:
                    with DEFAULT_KEYSTORE.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Exclude secret salts for secure public view
                    public_records = {}
                    for pid, entry in data.items():
                        public_records[pid] = {
                            "proof_id": entry.get("proof_id"),
                            "commitment_hash": entry.get("commitment_hash"),
                            "created_utc": entry.get("created_utc"),
                            "payload": entry.get("payload"),
                        }
                    self._send_json(200, public_records)
                    return
                except Exception as exc:
                    self._send_error_json(500, f"Error reading keystore: {exc}")
                    return
            else:
                self._send_error_json(404, "Keystore not found.")
                return

        # Serve static UI files
        rel_path = path.lstrip("/")
        if not rel_path or rel_path == "":
            rel_path = "index.html"

        file_path = (UI_DIR / rel_path).resolve()

        # Prevent directory traversal attacks
        if not str(file_path).startswith(str(UI_DIR.resolve())):
            self._send_error_json(403, "Forbidden")
            return

        if not file_path.is_file():
            # Fallback to index.html for SPA routing if requested file doesn't exist
            index_path = UI_DIR / "index.html"
            if index_path.is_file():
                file_path = index_path
            else:
                self._send_error_json(404, f"File not found: {rel_path}")
                return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            if file_path.suffix == ".css":
                mime_type = "text/css"
            elif file_path.suffix == ".js":
                mime_type = "application/javascript"
            elif file_path.suffix == ".json":
                mime_type = "application/json"
            else:
                mime_type = "application/octet-stream"

        try:
            with file_path.open("rb") as f:
                content = f.read()
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type or "json" in mime_type else mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            self._send_error_json(500, f"Error reading file: {exc}")

    def do_POST(self) -> None:
        """Route POST requests to engine runner and verification."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req_data = json.loads(body.decode("utf-8")) if body else {}
        except Exception as exc:
            self._send_error_json(400, f"Invalid JSON payload: {exc}")
            return

        # POST /api/run
        if path == "/api/run":
            try:
                from src.main import load_config, run_engine

                # Merge overrides into default config
                config_path = str(PROJECT_ROOT / "config.json")
                config = load_config(config_path, overrides=req_data.get("overrides", {}))

                # Run the engine
                report, artifact, exit_code = run_engine(config)

                # Save the new artifact if configured
                artifact_path = PROJECT_ROOT / config.get("output", {}).get("artifact_path", "artifacts/qtmce_result.json")
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                with artifact_path.open("w", encoding="utf-8") as f:
                    json.dump(artifact, f, indent=2)

                self._send_json(200, {
                    "success": exit_code == 0,
                    "exit_code": exit_code,
                    "report": report,
                    "artifact": artifact,
                })
            except Exception as exc:
                self._send_error_json(500, f"Execution failed: {exc}")
            return

        # POST /api/verify
        if path == "/api/verify":
            proof_id = req_data.get("proof_id")
            payload = req_data.get("payload")
            if not proof_id or payload is None:
                self._send_error_json(400, "Missing 'proof_id' or 'payload'")
                return

            try:
                scheme = LatticeCommitmentScheme(keystore_path=str(DEFAULT_KEYSTORE))
                verification = scheme.verify_detailed(proof_id, payload)
                self._send_json(200, verification)
            except Exception as exc:
                self._send_error_json(500, f"Verification failed: {exc}")
            return

        self._send_error_json(404, "Endpoint not found")

    def log_message(self, format: str, *args: Any) -> None:
        """Custom concise logging format."""
        sys.stderr.write(f"[BrahmaQ UI] {self.address_string()} - {format % args}\n")


def start_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Start and return the ThreadingHTTPServer instance."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, BrahmaQRequestHandler)
    print(f"============================================================")
    print(f"  BrahmaQ (Q-TMCE) Frontend Server is running!")
    print(f"  URL: http://{host}:{port}/")
    print(f"  API: http://{host}:{port}/api/status")
    print(f"============================================================")
    return httpd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BrahmaQ Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    httpd = start_server(host=args.host, port=args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()
