#!/usr/bin/env python3
"""
F1 Prediction API - Production Runner
Use this for production or production-like testing locally

Note: Gunicorn only works on Unix/Linux. On Windows, this script
will use waitress as an alternative, or fall back to Flask dev server.
"""

import subprocess
import sys
import os
import platform

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    port = os.getenv("PORT", "5000")
    workers = os.getenv("WORKERS", "2")
    
    print("🏎️  Starting F1 Prediction API (Production Mode)")
    print("=" * 50)
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"   Platform: {platform.system()}")
    print("=" * 50)
    
    if platform.system() == "Windows":
        # On Windows, try waitress first, then fall back to Flask
        try:
            import waitress
            print("   Server: Waitress (Windows)")
            print("=" * 50)
            from api import app
            waitress.serve(app, host="0.0.0.0", port=int(port), threads=int(workers) * 2)
        except ImportError:
            print("   Server: Flask Development (install waitress for production)")
            print("   Run: pip install waitress")
            print("=" * 50)
            from api import app
            app.run(host="0.0.0.0", port=int(port), debug=False, threaded=True)
    else:
        # On Unix/Linux, use Gunicorn
        print("   Server: Gunicorn (Unix/Linux)")
        print("=" * 50)
        cmd = [
            sys.executable, "-m", "gunicorn",
            "--bind", f"0.0.0.0:{port}",
            "--workers", workers,
            "--threads", "4",
            "--timeout", "120",
            "--access-logfile", "-",
            "--error-logfile", "-",
            "wsgi:app"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
