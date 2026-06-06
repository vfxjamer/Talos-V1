"""Build Talos on Kaggle. Run from the repo root."""
import os, subprocess, sys, re, threading
from pathlib import Path

LOCAL_DIR = os.environ.get("TALOS_DIR", os.getcwd())
BUILD_DIR = os.path.join(LOCAL_DIR, "build")
BINARY = "Talos"

def log(msg):
    print(f"\n═══ {msg} ═══", flush=True)

def run(cmd, **kw):
    print(f">>> {' '.join(cmd)}", flush=True)
    kw.setdefault("cwd", LOCAL_DIR)
    subprocess.run(cmd, check=True, **kw)

def apt_install(pkgs):
    run(["apt-get", "install", "-y", "-qq",
         "-oDPkg::Lock::Timeout=120", "--no-install-recommends"]
        + pkgs.split())

# ── Kill stale apt locks ──────────────────────────────────
for p in ["apt-get", "dpkg"]:
    subprocess.run(["pkill", "-9", p], capture_output=True)
for lock in ["/var/lib/dpkg/lock-frontend", "/var/lib/apt/lists/lock",
             "/var/cache/apt/archives/lock"]:
    try: os.remove(lock)
    except FileNotFoundError: pass
subprocess.run(["dpkg", "--configure", "-a"], capture_output=True)

# ── Build deps ─────────────────────────────────────────────
log("Installing build deps")
run(["apt-get", "update", "-qq", "-oDPkg::Lock::Timeout=120"])
apt_install("cmake build-essential pkg-config")

# ── LibTorch ───────────────────────────────────────────────
log("LibTorch")
libtorch_dir = os.path.join(LOCAL_DIR, "libtorch")
if not os.path.exists(libtorch_dir):
    apt_install("wget unzip")
    # Use cu117 to match Kaggle's default CUDA 11.5
    run(["wget", "-q", "--show-progress",
         "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu117.zip",
         "-O", "/tmp/libtorch.zip"])
    run(["unzip", "-q", "/tmp/libtorch.zip", "-d", LOCAL_DIR])
    os.remove("/tmp/libtorch.zip")
else:
    print("LibTorch already cached", flush=True)

# ── CUDA ───────────────────────────────────────────────────
log("CUDA")
cuda_found = False
# Kaggle has CUDA 11.5 at /usr by default
for d in ["/usr/local/cuda-11", "/usr/local/cuda", "/usr"]:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "bin/nvcc")):
        os.environ["CUDA_TOOLKIT_ROOT_DIR"] = d
        os.environ["PATH"] = f"{d}/bin:{os.environ.get('PATH', '')}"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print(f"Found CUDA at {d}", flush=True)
        cuda_found = True
        break

if not cuda_found:
    print("WARNING: CUDA not found!", flush=True)

# ── cmake configure ────────────────────────────────────────
log("cmake configure")
os.environ["CMAKE_PREFIX_PATH"] = libtorch_dir
result = subprocess.run(
    ["cmake", "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release"],
    cwd=LOCAL_DIR, capture_output=True, text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
if result.returncode != 0:
    raise subprocess.CalledProcessError(result.returncode, ["cmake", "-B", BUILD_DIR])

# ── cmake build ────────────────────────────────────────────
log("cmake build")
nproc = os.cpu_count() or 4
print(f"Building with {nproc} threads...", flush=True)

# Install tqdm for progress bar
try:
    import tqdm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"],
                   capture_output=True)
    import tqdm

# Parse cmake progress
proc = subprocess.Popen(
    ["cmake", "--build", BUILD_DIR, "--config", "Release", "-j", str(nproc)],
    cwd=LOCAL_DIR,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    universal_newlines=True, bufsize=1)

pct_pat = re.compile(r"^\[(\s*\d+)%\]")
bar = tqdm.tqdm(total=100, unit="%", desc="Compiling", ncols=80)

for line in proc.stdout:
    line = line.rstrip()
    m = pct_pat.match(line)
    if m:
        pct = int(m.group(1).strip())
        bar.n = pct
        bar.refresh()
    # Always print errors
    if "error:" in line.lower() or "warning:" in line.lower():
        print(line, flush=True)
    elif not m:
        print(line, flush=True)

bar.close()
proc.wait()
if proc.returncode != 0:
    raise subprocess.CalledProcessError(proc.returncode, ["cmake", "--build"])

log("Build complete")
