#!/usr/bin/env python3
"""
Cross-platform installer for MambaTalk.
Handles Linux/Windows/macOS differences automatically.

Usage:
    python install.py            # CUDA 12.1 (default)
    python install.py --cpu      # CPU-only (no CUDA)
    python install.py --cuda 118 # CUDA 11.8
"""
import sys
import os
import platform
import subprocess
import argparse


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", *args])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="CPU-only, no CUDA")
    parser.add_argument("--cuda", default="121", help="CUDA version tag, e.g. 118 or 121 (default: 121)")
    args = parser.parse_args()

    is_windows = platform.system() == "Windows"
    is_linux   = platform.system() == "Linux"
    is_mac     = platform.system() == "Darwin"
    py_ver     = sys.version_info
    py_tag     = f"cp{py_ver.major}{py_ver.minor}"

    print("=" * 60)
    print(f"  MambaTalk Installer")
    print(f"  OS      : {platform.system()} {platform.machine()}")
    print(f"  Python  : {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    print(f"  CUDA    : {'CPU only' if args.cpu else args.cuda}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. PyTorch
    # ------------------------------------------------------------------
    print("\n[1/5] Installing PyTorch 2.1.0")
    if args.cpu:
        pip("torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0")
    else:
        cuda_tag = f"cu{args.cuda}"
        pip("torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}")

    # ------------------------------------------------------------------
    # 2. Python requirements
    # ------------------------------------------------------------------
    print("\n[2/5] Installing Python requirements")
    pip("-r", "requirements.txt")

    # On Linux, also install pyvirtualdisplay (used for headless rendering)
    if is_linux:
        pip("pyvirtualdisplay==3.0")

    # ------------------------------------------------------------------
    # 3. causal_conv1d
    # ------------------------------------------------------------------
    print("\n[3/5] Installing causal_conv1d")
    if is_linux and py_tag == "cp39" and not args.cpu:
        # Official pre-built wheel (Linux x86_64, Python 3.9, CUDA 12.2)
        pip(
            "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/"
            "causal_conv1d-1.4.0+cu122torch2.1cxx11abiTRUE-cp39-cp39-linux_x86_64.whl"
        )
    else:
        # Build from source — requires CUDA toolkit + compiler (MSVC on Windows, gcc on Linux/Mac)
        print(f"  Note: building causal_conv1d from source for {platform.system()} / Python {py_ver.major}.{py_ver.minor}")
        if is_windows:
            print("  Windows: ensure Visual Studio Build Tools and CUDA Toolkit are installed.")
        pip("causal-conv1d>=1.1.0", "--no-build-isolation")

    # ------------------------------------------------------------------
    # 4. mamba_ssm
    # ------------------------------------------------------------------
    print("\n[4/5] Installing mamba_ssm")
    if is_linux and py_tag == "cp39" and not args.cpu:
        # Official pre-built wheel (Linux x86_64, Python 3.9, CUDA 11)
        pip(
            "https://github.com/state-spaces/mamba/releases/download/v2.2.4/"
            "mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
        )
    else:
        print(f"  Note: building mamba_ssm from source for {platform.system()} / Python {py_ver.major}.{py_ver.minor}")
        if is_windows:
            print("  Windows: ensure Visual Studio Build Tools and CUDA Toolkit are installed.")
        pip("mamba-ssm", "--no-build-isolation")

    # ------------------------------------------------------------------
    # 5. PyTorch3D  (Linux only — no Windows wheel exists)
    # ------------------------------------------------------------------
    print("\n[5/5] PyTorch3D")
    if is_linux:
        print("  Installing PyTorch3D from source...")
        pip("git+https://github.com/facebookresearch/pytorch3d.git@stable")
    else:
        print(f"  Skipping PyTorch3D — no pre-built wheel for {platform.system()}.")
        print("  3D body rendering (render.py) will be unavailable.")
        print("  Training and inference are not affected.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Installation complete!")
    print("=" * 60)

    if is_windows:
        print("""
Windows notes:
  - Training and inference work normally.
  - 3D body rendering (render.py) requires pyrender which is already
    installed. It uses OpenGL via pyglet — no virtual display needed.
  - Run scripts: use run_scripts\\test.bat and run_scripts\\train.bat
    instead of the .sh versions.
  - ffmpeg must be installed and on PATH:
      winget install Gyan.FFmpeg
    or download from https://ffmpeg.org/download.html
""")
    elif is_linux:
        print("""
Linux notes:
  - For headless rendering use: xvfb-run -a python render.py ...
  - ffmpeg: conda install -c conda-forge ffmpeg
""")


if __name__ == "__main__":
    main()
