#!/usr/bin/env python3
"""
Unified Pipeline Runner for Chat & Relationship Analytics
Single-command CLI to parse, analyze, generate reports, and launch the dashboard.

Usage:
    python run.py                   # Run full pipeline and launch dashboard
    python run.py --parse           # Parse raw chat file
    python run.py --analyze         # Run NLP and behavioral analysis
    python run.py --report          # Print relationship summary report
    python run.py --dashboard       # Launch dashboard only
    python run.py --input custom.txt # Use custom chat export file
"""

import os
import sys
import argparse
import webbrowser
import threading
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def log(msg: str):
    print(f"[pipeline] {msg}")


def run_command(cmd_args, desc: str):
    log(f"▶ {desc}...")
    result = subprocess.run(cmd_args, cwd=BASE_DIR)
    if result.returncode != 0:
        log(f"❌ Error during {desc}. Process exited with code {result.returncode}")
        sys.exit(result.returncode)
    log(f"✔ Completed {desc}.")


def main():
    parser = argparse.ArgumentParser(description="Chat & Relationship Analytics Pipeline")
    parser.add_argument('-i', '--input', default='chat.txt', help="Path to raw chat export (default: chat.txt)")
    parser.add_argument('--parse', action='store_true', help="Run parser step only")
    parser.add_argument('--analyze', action='store_true', help="Run analysis step only")
    parser.add_argument('--report', action='store_true', help="Print relationship report only")
    parser.add_argument('--dashboard', action='store_true', help="Launch dashboard only")
    parser.add_argument('--port', type=int, default=8050, help="Port for dashboard (default: 8050)")
    parser.add_argument('--no-browser', action='store_true', help="Do not automatically open web browser")

    args = parser.parse_args()

    python_bin = sys.executable

    # If no specific stage flag is set, run the complete end-to-end pipeline
    run_all = not (args.parse or args.analyze or args.report or args.dashboard)

    # 1. Parse Step
    if run_all or args.parse:
        run_command([python_bin, "conversation_parser.py", args.input, "-o", "chat.parsed.csv"], "Chat Parsing")

    # 2. Analyze Step
    if run_all or args.analyze:
        run_command([python_bin, "conversation_analysis.py", "chat.parsed.csv", "-o", "chat.parsed.analyzed.csv"], "NLP & Behavioral Analysis")

    # 3. Report Step
    if run_all or args.report:
        run_command([python_bin, "relationship_analysis.py", "chat.parsed.analyzed.csv"], "Relationship Summary Report")

    # 4. Dashboard Step
    if run_all or args.dashboard:
        url = f"http://127.0.0.1:{args.port}/"
        if not args.no_browser:
            def open_browser():
                import time
                time.sleep(1.2)
                webbrowser.open(url)
            threading.Thread(target=open_browser, daemon=True).start()

        os.environ['PORT'] = str(args.port)
        log(f"🌐 Launching Interactive Dashboard at {url}")
        subprocess.run([python_bin, "conversation_dashboard.py"], cwd=BASE_DIR)


if __name__ == '__main__':
    main()
