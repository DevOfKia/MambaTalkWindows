@echo off
set GPU=0
set CUDA_VISIBLE_DEVICES=%GPU%

python -c "import random; print(random.randint(8600,8800))" > %TEMP%\port.txt
set /p PORT=<%TEMP%\port.txt
del %TEMP%\port.txt

python train.py ^
    --config configs/mambatalk.yaml ^
    --model mambatalk ^
    --test_start_epoch 80 ^
    --test_period 5 ^
    --port %PORT%
