# -*- coding: utf-8 -*-
# ==========================================================
#  个人知识库 —— 一键完整重建脚本
#  适用：rag_storage 因【进程被强杀 / 多个实例并发启动写盘】而被写坏时，
#        用一份干净完整的数据重建（图谱、doc_status、全文、向量、缓存）。
#
#  重要：必须在【你自己的终端】里运行。
#        不要在本会话/自动化宿主里跑——宿主会周期性强杀后台服务，
#        每次强杀都可能让 LightRAG 写回不完整快照而损坏 rag_storage。
#
#  用法（在 LightRAG_test 目录）：
#     powershell -ExecutionPolicy Bypass -File .\rebuild_kb.ps1
#     powershell -ExecutionPolicy Bypass -File .\rebuild_kb.ps1 -Reset
#     powershell -ExecutionPolicy Bypass -File .\rebuild_kb.ps1 -Reset -Keep
#
#    -Reset  重建前把现有 rag_storage 改名备份（不删除），从空目录开始。
#    -Keep   处理完后服务器保持常驻（不自动关闭），供 WebUI 使用。
# ==========================================================
param(
    [switch]$Reset,
    [switch]$Keep
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot

Write-Host "=== 个人知识库重建 ===" -ForegroundColor Cyan

if (-not (Test-Path ".env")) {
    Write-Host "[错误] 缺少 .env（LLM/Embedding 配置）。参考 env.example。" -ForegroundColor Red
    exit 1
}

$hasServer = @(Get-NetTCPConnection -LocalPort 9621 -State Listen -ErrorAction SilentlyContinue).Count -gt 0

# ---- Reset：备份旧存储（必须在启动服务器之前做）----
if ($Reset) {
    if ($hasServer) {
        Write-Host "[错误] -Reset 前请先停止正在 9621 上跑的旧服务器（否则新服务器仍会读到旧存储）。" -ForegroundColor Red
        Write-Host "       可执行: Get-NetTCPConnection -LocalPort 9621 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
        exit 3
    }
    if (Test-Path "rag_storage") {
        $bk = "rag_storage_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item "rag_storage" $bk
        Write-Host "[1/5] 旧存储已备份到 $bk（从空目录重建）" -ForegroundColor Yellow
    } else {
        Write-Host "[1/5] 无旧存储，直接以空目录重建。"
    }
} else {
    Write-Host "[1/5] 保留现有存储，只会把缺失的源文档追加进去。"
}

# ---- 启动服务器 ----
if ($hasServer) {
    Write-Host "[2/5] 检测到 9621 已有服务在跑，直接复用它。" -ForegroundColor Yellow
} else {
    Write-Host "[2/5] 启动 LightRAG Server（独立进程，日志见 _rebuild_out/err.log）"
    $srv = Start-Process -FilePath "D:\anaconda\python.exe" `
        -ArgumentList "-u","-m","lightrag.api.lightrag_server" `
        -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput (Join-Path $PSScriptRoot "_rebuild_out.log") `
        -RedirectStandardError (Join-Path $PSScriptRoot "_rebuild_err.log") `
        -WindowStyle Hidden -PassThru
    Set-Content (Join-Path $PSScriptRoot "_server.pid") $srv.Id
    Write-Host ("    服务器 PID={0}，等待就绪..." -f $srv.Id)
    $up = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 900
        try { $null = Invoke-RestMethod "http://127.0.0.1:9621/health" -TimeoutSec 3; $up = $true; break } catch {}
    }
    if (-not $up) {
        Write-Host "[错误] 服务器未就绪。看 _rebuild_err.log。" -ForegroundColor Red
        exit 2
    }
    Write-Host "    服务器已就绪。"
}

# ---- 上传源文档 ----
Write-Host "[3/5] 上传源文档（OCR 抽取结果 + 飞书样例 docx）"
$src = @()
Get-ChildItem "lightrag\api\routers\output\*.md" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "*_ocr_result_ocr_result*" } |
    ForEach-Object { $src += $_.FullName }
$docx = "inputs\__enqueued__\_测试_飞书导入规范.docx"
if (Test-Path $docx) { $src += (Resolve-Path $docx).Path }

Write-Host ("    共 {0} 个源文件" -f $src.Count)
foreach ($f in $src) {
    python kb_tools.py upload $f | Out-Null
}
Write-Host "    上传完毕，开始后台处理。"

# ---- 等待处理完成 ----
Write-Host "[4/5] 等待全部处理完成（期间不要关本窗口，服务器被杀会自动重试）"
$env:PYTHONIOENCODING = "utf-8"
python kb_tools.py wait --timeout 3600

# ---- 最终状态 ----
Write-Host "[5/5] 最终文档状态"
python kb_tools.py status

# ---- 关闭还是保留 ----
if (-not $Keep -and -not $hasServer) {
    $pid = Get-Content "_server.pid" -ErrorAction SilentlyContinue
    if ($pid) { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue; Write-Host "服务器已关闭（处理完成）。" }
}
Write-Host "=== 完成。WebUI: http://localhost:9621（若 -Keep 或自启则常驻）===" -ForegroundColor Green
