"""Run both local entry points, allowing the camera writer to flush on exit."""
import os
import subprocess
import sys

from .run import main

if __name__=="__main__":
    if "--db" in sys.argv:
        os.environ["DROWSINESS_DB"] = sys.argv[sys.argv.index("--db")+1]
    import fastapi
    dashboard = subprocess.Popen([sys.executable,"-m","uvicorn","monitor.api:app","--host","127.0.0.1","--port","8000"])
    try:
        main()
    finally:
        dashboard.terminate()
        dashboard.wait()
