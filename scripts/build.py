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
# Prefer /usr/local/cuda (CUDA 13.3 on Kaggle) over /usr/lib/nvidia-cuda-toolkit (broken stub)
for d in ["/usr/local/cuda-12", "/usr/local/cuda", "/usr/local/cuda-11"]:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "bin/nvcc")):
        os.environ["CUDA_TOOLKIT_ROOT_DIR"] = d
        # Put this CUDA first in PATH to override the old broken nvidia-cuda-toolkit
        os.environ["PATH"] = f"{d}/bin:{os.environ.get('PATH', '')}"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Force ptxas and nvcc to use the newer versions
        if os.path.isfile(f"{d}/bin/ptxas"):
            os.environ["CUDA_PTXAS_EXECUTABLE"] = f"{d}/bin/ptxas"
        if os.path.isfile(f"{d}/bin/nvcc"):
            os.environ["CMAKE_CUDA_COMPILER"] = f"{d}/bin/nvcc"
        # Remove /usr/bin/nvcc symlink that points to broken stub
        # This forces cmake to use our specified CMAKE_CUDA_COMPILER
        if os.path.islink("/usr/bin/nvcc") or os.path.exists("/usr/bin/nvcc"):
            try:
                os.remove("/usr/bin/nvcc")
                print("Removed /usr/bin/nvcc stub", flush=True)
            except:
                pass
        # Print nvcc version
        nvcc_ver = subprocess.run([f"{d}/bin/nvcc", "--version"], capture_output=True, text=True)
        print(f"Found CUDA at {d}", flush=True)
        print(f"nvcc version: {nvcc_ver.stdout.strip()}", flush=True)
        cuda_found = True
        break

if not cuda_found:
    print("WARNING: CUDA not found!", flush=True)

# ── NVTX (NVIDIA Tools Extension) ─────────────────────────
log("NVTX")
cuda_root = os.environ.get("CUDA_TOOLKIT_ROOT_DIR", "/usr/local/cuda")
# CUDA 13.x has NVTX v3 at include/nvtx3/ but libtorch's cmake looks for old-style nvToolsExt.h
# Create symlink from new location to where libtorch expects it
import glob as _glob
nvtx3_dir = f"{cuda_root}/include/nvtx3"
if os.path.isdir(nvtx3_dir):
    print(f"Found NVTX v3 headers at {nvtx3_dir}", flush=True)
    # Find the actual nvToolsExt.h (might be in a subdir like nvtx3/ or nvtx3/detail/)
    candidates = _glob.glob(f"{nvtx3_dir}/**/nvToolsExt.h", recursive=True)
    if candidates:
        actual_header = candidates[0]
        # Create symlink at expected location: /usr/local/cuda/include/nvtx3/nvToolsExt.h
        expected_path = f"{nvtx3_dir}/nvToolsExt.h"
        if not os.path.isfile(expected_path):
            os.symlink(actual_header, expected_path)
            print(f"Created symlink: {expected_path} -> {actual_header}", flush=True)
    else:
        # List what's there
        all_headers = _glob.glob(f"{nvtx3_dir}/**/*.h", recursive=True)
        print(f"NVTX headers available: {all_headers[:10]}", flush=True)
else:
    print(f"WARNING: NVTX v3 not found at {nvtx3_dir}", flush=True)

# ── cmake configure ────────────────────────────────────────
log("cmake configure")
# Clean any previous failed cmake cache
import shutil as _shutil
if os.path.exists(BUILD_DIR):
    print(f"Cleaning previous build dir: {BUILD_DIR}")
    _shutil.rmtree(BUILD_DIR, ignore_errors=True)
os.environ["CMAKE_PREFIX_PATH"] = libtorch_dir
# Set CUDA arch list for T4 (sm_75) to skip detection test
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
# Print which nvcc is being used
print(f"PATH: {os.environ['PATH']}", flush=True)
print(f"which nvcc: {subprocess.run(['which', 'nvcc'], capture_output=True, text=True).stdout.strip()}", flush=True)
print(f"nvcc --version: {subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout.strip()}", flush=True)
print(f"ptxas --version: {subprocess.run(['ptxas', '--version'], capture_output=True, text=True).stderr.strip()}", flush=True)
result = subprocess.run(
    ["cmake", "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release",
     "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
     "-DCMAKE_CUDA_FLAGS=-std=c++17",
     "-DCMAKE_CUDA_STANDARD=17",
     "-DCMAKE_CUDA_ARCHITECTURES=75",
     "-DTORCH_CUDA_ARCH_LIST=7.5"],
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
