# Prints the hardware and software of a Kaggle T4 session (same accelerator as the E1 runs).
import os, platform, subprocess, sys
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
print("=== CPU"); print(sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core\\(s\\)|Socket'")); print("nproc:", os.cpu_count())
print("=== RAM"); print(sh("free -g")); print(sh("grep MemTotal /proc/meminfo"))
print("=== OS"); print(sh("cat /etc/os-release | grep -E 'PRETTY_NAME'")); print("kernel:", platform.release())
print("=== GPU"); print(sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"))
print("=== DISK"); print(sh("df -h /kaggle/working | tail -1"))
print("=== SOFTWARE"); print("python", sys.version.split()[0])
try:
    import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
except Exception as e: print("torch ERR", e)
print(sh("pip show ultralytics 2>/dev/null | grep Version") or "ultralytics not preinstalled")
