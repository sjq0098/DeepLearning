# ============================================================================
#  First exploratory pass of the edge-sharpening improvements, run sequentially.
#  Each variant = A5 full-PPA baseline + ONE change (clean isolation vs A5):
#     A6 = + BAS boundary head            (③)
#     A7 = + Boundary-IoU + SSIM loss     (①)
#     A8 = + image-edge consistency loss  (A8)
#  200 epochs / batch 16 to match the A1-A5 ablations. Weights are first-pass
#  defaults; tune later. Metrics (incl. Boundary MAE/F) -> ./improve_logs\.
# ============================================================================
$Python = "C:\Users\admin\.conda\envs\dl\python.exe"
$Data   = ".\data"
$Epochs = 200
$Batch  = 16

New-Item -ItemType Directory -Force -Path ".\improve_logs" | Out-Null

# name -> @(train extra flags, eval extra flags)
$Runs = @(
    @{ name = "A6"; train = "--use_bas";                              eval = "--use_bas" },
    @{ name = "A7"; train = "--boundary_weight 1.0 --ssim_weight 1.0"; eval = "" },
    @{ name = "A8"; train = "--edge_weight 10.0";                      eval = "" }
)

foreach ($r in $Runs) {
    $name = $r.name
    $save = ".\checkpoints_ablation\$name"
    $ckpt = "$save\model_epoch${Epochs}.pth"
    Write-Host "================ TRAIN $name ($Epochs ep) ================" -ForegroundColor Cyan

    $trainArgs = "train.py --datapath $Data --train_split train --loss ppa " +
                 "--epochs $Epochs --batch_size $Batch --workers 4 --save_every 100 " +
                 "--savepath $save " + $r.train
    Start-Process -FilePath $Python -ArgumentList $trainArgs -NoNewWindow -Wait

    Write-Host "================ EVAL  $name ================" -ForegroundColor Green
    $log = ".\improve_logs\$name.txt"
    $evalArgs = "evaluate.py --checkpoint $ckpt --datapath $Data --workers 0 " + $r.eval
    & $Python $evalArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries) *>&1 | Tee-Object -FilePath $log
    Write-Host "  -> metrics saved to $log" -ForegroundColor Green
}

Write-Host "`nAll three runs complete. Compare against A5 baseline:" -ForegroundColor Yellow
Write-Host "  A5  MAE 0.0435 | F_mean 0.9010 | Boundary MAE 0.1751 | Boundary F 0.8375" -ForegroundColor Yellow
