@echo off
REM ===========================================================================
REM task3 - Run all three loss-ablation experiments and the comparison plot.
REM ===========================================================================
setlocal

@REM echo [task3] Running CE-only experiment ...
@REM python train.py --config configs/baseline_ce.yaml
@REM if errorlevel 1 goto :error

echo [task3] Running Dice-only experiment ...
python train.py --config configs/baseline_dice.yaml
if errorlevel 1 goto :error

echo [task3] Running CE+Dice experiment ...
python train.py --config configs/baseline_ce_dice.yaml
if errorlevel 1 goto :error

echo [task3] Generating comparison figures ...
python experiments/compare.py
if errorlevel 1 goto :error

echo [task3] All experiments completed successfully.
exit /b 0

:error
echo [task3] An experiment failed. Aborting.
exit /b 1
