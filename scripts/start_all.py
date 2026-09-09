import subprocess
import sys
import time

commands = [
    # 1. JupyterLab
    ("JupyterLab", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"),
    # 2. FastAPI
    ("FastAPI", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"),
    # 3. Streamlit (con headless para que no pida email)
    ("Streamlit", "streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"),
    # 4. React Frontend
    ("Frontend (Vite)", "cd frontend && pnpm dev --host 0.0.0.0")
]

processes = []

print("🚀 Iniciando ecosistema EuroSAT AI Lab...\n")

try:
    for name, cmd in commands:
        print(f"-> Lanzando {name}...")
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)
        time.sleep(1.5)

    print("\n✅ Todos los servicios están en ejecución:")
    print(" - JupyterLab:  http://localhost:8888")
    print(" - FastAPI API: http://localhost:8000/docs")
    print(" - Streamlit:   http://localhost:8501")
    print(" - React/Vite:  http://localhost:5173\n")
    print("Presiona Ctrl+C para detenerlos todos.")

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Deteniendo todos los servicios...")
    for p in processes:
        p.terminate()
    sys.exit(0)
