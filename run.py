"""Launch the Sapiens Playground web app, auto-selecting a free port.

Tries the preferred port (default 8000) and, if it's busy, scans upward for
the next free one so the app always starts.

Usage:
    python run.py                  # try 8000, else next free port
    python run.py --port 8080      # try 8080 first
    python run.py --host 0.0.0.0   # bind all interfaces (LAN access)
    python run.py --reload         # dev auto-reload
    python run.py --normal-size 1b # use the heavier 1B normals model (0.6b/1b/2b)
"""
import argparse
import os
import socket
import uvicorn


def is_free(host: str, port: int) -> bool:
    """True if we can bind host:port right now (i.e. nothing else is using it)."""
    # No SO_REUSEADDR: we want the bind to FAIL when a server is already running.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        if is_free(host, port):
            return port
    raise SystemExit(
        f"No free port found in {preferred}-{preferred + attempts - 1}. "
        f"Free one up or pass --port.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Sapiens Playground web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000, help="preferred port (default 8000)")
    ap.add_argument("--reload", action="store_true", help="enable dev auto-reload")
    ap.add_argument("--normal-size", choices=["0.6b", "1b", "2b"],
                    help="normals model size (default 0.6b; 2b runs in fp16 to fit 8GB)")
    args = ap.parse_args()

    # Must be set before the app imports sapiens_infer / download_models.
    if args.normal_size:
        os.environ["SAPIENS_NORMAL_SIZE"] = args.normal_size

    port = find_free_port(args.host, args.port)
    if port != args.port:
        print(f"[run] Port {args.port} is busy -> using {port} instead")
    print(f"[run] Sapiens Playground -> http://{args.host}:{port}\n")

    uvicorn.run("app:app", host=args.host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
