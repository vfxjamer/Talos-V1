"""Build Talos on Kaggle. Run from the repo root (LOCAL_DIR)."""
import os, subprocess, sys

LOCAL_DIR = os.environ.get("TALOS_DIR", os.getcwd())
BUILD_DIR = os.path.join(LOCAL_DIR, "build")
BINARY = "Talos"

def log(msg):
    print(f"═══ {msg} ═══", flush=True)

def run(cmd, **kw):
    print(f">>> {' '.join(cmd)}", flush=True)
    kw.setdefault("cwd", LOCAL_DIR)
    subprocess.run(cmd, check=True, **kw)

def apt_install(pkgs):
    run(["apt-get", "install", "-y", "-qq",
         "-oDPkg::Lock::Timeout=120", "--no-install-recommends"]
        + pkgs.split())

# ── Kill stale apt locks from previous killed sessions ─────
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
print("OK", flush=True)

# ── LibTorch ───────────────────────────────────────────────
log("LibTorch")
if not os.path.exists(os.path.join(LOCAL_DIR, "libtorch")):
    print("Downloading LibTorch (~2GB)...", flush=True)
    apt_install("wget unzip")
    run(["wget", "-q", "--show-progress",
         "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip",
         "-O", "/tmp/libtorch.zip"])
    run(["unzip", "-q", "/tmp/libtorch.zip", "-d", LOCAL_DIR])
    os.remove("/tmp/libtorch.zip")
    print("LibTorch extracted", flush=True)
else:
    print("LibTorch already cached", flush=True)

# ── CUDA ───────────────────────────────────────────────────
log("CUDA")
for d in ["/usr/local/cuda-12", "/usr/local/cuda", "/usr/local/cuda-12.1"]:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "bin/nvcc")):
        os.environ["CUDA_TOOLKIT_ROOT_DIR"] = d
        os.environ["PATH"] = f"{d}/bin:{os.environ.get('PATH', '')}"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print(f"Found CUDA at {d}", flush=True)
        break
else:
    print("WARNING: CUDA 12 not found!", flush=True)

# ── cmake configure ────────────────────────────────────────
log("cmake configure")
os.environ["CMAKE_PREFIX_PATH"] = os.path.join(LOCAL_DIR, "libtorch")
has_build = os.path.isfile(os.path.join(BUILD_DIR, "build.ninja"))
has_build = has_build or os.path.isfile(os.path.join(BUILD_DIR, "Makefile"))
run(["cmake", "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release"])

# ── cmake build ────────────────────────────────────────────
log("cmake build")
nproc = os.cpu_count() or 4
run(["cmake", "--build", BUILD_DIR, "--config", "Release", "-j", str(nproc)])

log("Build complete")
