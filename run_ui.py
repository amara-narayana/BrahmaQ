#!/usr/bin/env python3
"""Convenient launcher for the BrahmaQ (Q-TMCE) Frontend UI.

Starts the local API server and automatically opens the dashboard in your default browser.
Usage:
    python run_ui.py
    python run_ui.py --port 8080 --no-browser
"""

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.server import start_server


def open_browser_delayed(url: str, delay: float = 1.0) -> None:
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch BrahmaQ UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"

    if not args.no_browser:
        t = threading.Thread(target=open_browser_delayed, args=(url,), daemon=True)
        t.start()

    httpd = start_server(host=args.host, port=args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BrahmaQ UI server...")
        httpd.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
