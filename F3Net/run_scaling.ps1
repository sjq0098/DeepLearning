# ============================================================================
#  ② "Architecture is data-hungry" sweep
#  Trains A1 (bone: no CFM / N=1 / no MLS) vs A4 (full arch: CFM+CFD+MLS),
#  both under BCE loss, on nested subsets {175,350,525,700}. The gap
#  (A4 metric - A1 metric) at each size shows whether the architectural
#  inductive bias only starts paying off with more data.
#
#  Run prepare_subsets.py first. Results -> ./scaling_logs/<variant>_<n>.txt
#  Each run is ~independent; ~8 trainings total. Adjust $Epochs to trade
#  rigor for wall-clock (baseline used 200). Fixed-epochs protocol: smaller
#  subsets see fewer gradient steps — note this when reading the curve.
# ============================================================================
$Python = "C:\Users\admin\.conda\envs\dl\python.exe"
$Sizes  = @(175, 350, 525, 700)
$Epochs = 200
$Batch  = 16
$Data   = ".\data"

New-Item -ItemType Directory -Force -Path ".\scaling_logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".\checkpoints_scaling" | Out-Null

# variant name -> extra train.py flags (A4 uses defaults = full arch)
$Variants = @{
    "A1bone" = "--no_cfm --num_decoders 1 --no_mls"
    "A4full" = "--num_decoders 2"
}
# eval flags must match how each variant was trained
$EvalFlags = @{
    "A1bone" = "--no_cfm --num_decoders 1 --no_mls"
    "A4full" = "--num_decoders 2"
}

foreach ($n in $Sizes) {
    foreach ($v in @("A1bone", "A4full")) {
        $save = ".\checkpoints_scaling\${v}_${n}"
        $ckpt = "$save\model_epoch${Epochs}.pth"
        Write-Host "=== TRAIN $v on $n images ($Epochs epochs) ===" -ForegroundColor Cyan

        $trainArgs = "train.py --datapath $Data --train_split train_$n --savepath $save " +
                     "--loss bce --epochs $Epochs --batch_size $Batch --save_every 0 " +
                     $Variants[$v]
        Start-Process -FilePath $Python -ArgumentList $trainArgs -NoNewWindow -Wait

        Write-Host "=== EVAL $v _ $n ===" -ForegroundColor Green
        $log = ".\scaling_logs\${v}_${n}.txt"
        $evalArgs = "evaluate.py --checkpoint $ckpt --datapath $Data --workers 0 " + $EvalFlags[$v]
        & $Python $evalArgs.Split(" ") *>&1 | Tee-Object -FilePath $log
    }
}

Write-Host "`nAll runs done. Metrics in .\scaling_logs\  ->  plot (A4full - A1bone) vs size." -ForegroundColor Yellow
