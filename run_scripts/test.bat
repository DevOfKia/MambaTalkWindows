@echo off
set GPU=0
set CUDA_VISIBLE_DEVICES=%GPU%
python test.py --config configs/mambatalk.yaml
