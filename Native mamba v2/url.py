import subprocess
import sys
import time

# 1. Install pyngrok if not already installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])

from pyngrok import ngrok

# 2. Configure ngrok authtoken
NGROK_TOKEN = "3HDwJmMvQ5yoXUBSSBguE1xx9yO_2ZTyiSLVcbhK8vunC9zJM"
ngrok.set_auth_token(NGROK_TOKEN)

# 3. Terminate any previous tunnels to avoid duplicate port errors
try:
    ngrok.kill()
except Exception:
    pass

# 4. Open ngrok tunnel on port 8888
tunnel = ngrok.connect(8888)
connect_url = f"{tunnel.public_url}/?token=kaggle123"

print("\n" + "=" * 65)
print(">>> KAGGLE JUPYTER KERNEL SERVER IS READY! <<<")
print("\nCopy this URL and paste it into your IDE 'Existing Jupyter Server':")
print(f"\n👉  {connect_url}  👈\n")
print("=" * 65 + "\n")

# 5. Start Jupyter Notebook server in Kaggle
cmd = [
    "jupyter", "notebook",
    "--ip=0.0.0.0",
    "--port=8888",
    "--no-browser",
    "--allow-root",
    "--NotebookApp.token=kaggle123",
    "--NotebookApp.allow_origin=*",
    "--NotebookApp.disable_check_xsrf=True"
]

try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("Server stopped.")
