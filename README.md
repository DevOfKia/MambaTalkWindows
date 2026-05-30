# MambaTalk Windows
<img width="1270" height="392" alt="Screenshot 2026-05-30 194302" src="https://github.com/user-attachments/assets/ab2fc7d0-b734-4977-9c92-49633595480a" />

> Official PyTorch implementation of [MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models](https://arxiv.org/pdf/2403.09471) (NeurIPS 2024).

MambaTalk uses Selective State Space Models to generate efficient, high-quality holistic human gestures from speech audio.

**This fork adds full Windows support and a one-command installer.**

---

## Quick Start

### 1. Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.9 recommended | 3.10–3.11 work but require building Mamba from source |
| CUDA | 12.1 | Must match PyTorch wheel |
| Conda | any | Optional but recommended for env isolation |
| ffmpeg | any | Required for video output |

**Install ffmpeg:**
- **Windows:** `winget install Gyan.FFmpeg` (or download from [ffmpeg.org](https://ffmpeg.org/download.html))
- **Linux/Ubuntu:** `sudo apt install ffmpeg`  
- **macOS:** `brew install ffmpeg`

**Windows only — extra build tools** (only needed if on Python ≠ 3.9):
- [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with **Desktop development with C++**
- [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)

---

### 2. Create Conda Environment (Recommended)

```bash
conda create -n mambatalk python=3.9.21
conda activate mambatalk
```

---

### 3. Install All Dependencies (One Command)

```bash
python install.py
```

That's it. The installer handles OS detection and installs the correct packages automatically.

**Options:**

```bash
python install.py --cuda 118   # CUDA 11.8 instead of 12.1
python install.py --cpu        # CPU-only (no CUDA, for testing only)
```

What the installer does per platform:

| Step | Linux (Python 3.9) | Linux (other Python) | Windows |
|------|-------------------|---------------------|---------|
| PyTorch | Official wheel | Official wheel | Official wheel |
| `causal_conv1d` | Official pre-built | Build from source | Build from source |
| `mamba_ssm` | Official pre-built | Build from source | Build from source |
| `pytorch3d` | Build from source | Build from source | **Skipped** (not needed for inference) |
| `pyvirtualdisplay` | Installed | Installed | **Skipped** (not needed on Windows) |

---

### 4. Download Pretrained Weights

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download --resume-download kkakkkka/MambaTalk --local-dir pretrained
```

Expected structure:

```
pretrained/
├── pretrained_vq/
│   ├── face.bin
│   ├── foot.bin
│   ├── hands.bin
│   ├── lower_foot.bin
│   └── upper.bin
├── smplx_models/
│   └── smplx/
│       └── SMPLX_NEUTRAL_2020.npz
├── test_sequences/
└── mambatalk_100.bin
```

---

## Examples & Demos

Here is MambaTalk generating holistic gestures (face, hands, body) from raw speech audio in real-time.

<p align="center">
  <video src="https://github.com/user-attachments/assets/8fa9fa27-56c2-4000-afcd-bd580eb3cbce" autoplay loop muted playsinline width="85%">
      
  </video>
</p>

> 💡 **Note:** If you want to test your own audio files, check out the [Running Inference](#running-inference) section below.

---
## Running Inference

### 1. Download the BEAT2 Dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/H-Liu1997/BEAT2
```

### 2. Run Evaluation

**Linux / macOS:**
```bash
bash run_scripts/test.sh
```

**Windows:**
```bat
run_scripts\test.bat
```

Or directly:
```bash
python test.py --config configs/mambatalk.yaml
```

### 3. Visualize Results (render to video)

**Linux (headless):**
```bash
xvfb-run -a python render.py \
    --npy_path ./res_2_scott_0_1_1.npz \
    --wav_path ./BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_1_1.wav \
    --save_dir outputs/render
```

**Windows / Linux with display:**
```bash
python render.py \
    --npy_path ./res_2_scott_0_1_1.npz \
    --wav_path ./BEAT2/beat_english_v2.0.0/wave16k/2_scott_0_1_1.wav \
    --save_dir outputs/render
```

---

## Training

**Linux / macOS:**
```bash
bash run_scripts/train.sh
```

**Windows:**
```bat
run_scripts\train.bat
```

### Train Individual VQ-VAE Components

```bash
python train.py --config configs/cnn_vqvae_face_30.yaml      # Face
python train.py --config configs/cnn_vqvae_hands_30.yaml     # Hands
python train.py --config configs/cnn_vqvae_lower_30.yaml     # Lower Body
python train.py --config configs/cnn_vqvae_lower_foot_30.yaml # Lower Foot
python train.py --config configs/cnn_vqvae_upper_30.yaml     # Upper Body
```

---

## Custom Data

Organize your data like this:

```
your_data/
├── smplxflame_30/
│   ├── 2_scott_0_1_1.npz
│   └── 2_scott_0_2_2.npz
├── test.csv
├── textgrid/
│   ├── 2_scott_0_1_1.TextGrid
│   └── 2_scott_0_2_2.TextGrid
└── wave16k/
    ├── 2_scott_0_1_1.wav
    └── 2_scott_0_2_2.wav
```

`test.csv` format:
```csv
id,type
2_scott_0_1_1,test
2_scott_0_2_2,test
```

### Generate TextGrid Files (Audio Alignment)

```bash
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
conda install -c conda-forge kalpy
pip install pgvector Bio

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align ./data english_us_arpa english_us_arpa ./data/result
```

---

## Troubleshooting

### `mamba_ssm` build fails on Windows

You need:
1. **Visual Studio Build Tools 2022** with "Desktop development with C++" workload
2. **CUDA Toolkit** matching your PyTorch CUDA version
3. **Ninja:** `pip install ninja`

Then retry: `python install.py`

### `pyrender` fails on headless Linux

Install Xvfb and use `xvfb-run`:
```bash
sudo apt install xvfb
xvfb-run -a python render.py ...
```

### `ffmpeg not found`

ffmpeg must be on your system PATH. Test with: `ffmpeg -version`

### Out of GPU memory

Reduce `batch_size` in `configs/mambatalk.yaml`.

---

## Citation

```bibtex
@article{xu2024mambatalk,
  title={Mambatalk: Efficient holistic gesture synthesis with selective state space models},
  author={Xu, Zunnan and Lin, Yukang and Han, Haonan and Yang, Sicheng and Li, Ronghui and Zhang, Yachao and Li, Xiu},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={20055--20080},
  year={2024}
}
```
