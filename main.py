import subprocess
import sys

torch = "handwritten_torch.py"
normal = "handwritten.py"

if len(sys.argv) > 1:
    print("Training New Model")
    chosen = normal
else:
    print("Using Torch Version")
    chosen = torch

try:
    subprocess.run(["python", chosen], check=True)
except:
    print("error")
