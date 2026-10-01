# -*- coding: utf-8 -*-
# 一键启动个人知识库（LightRAG Server + WebUI）
# 用法: powershell -ExecutionPolicy Bypass -File .\start_kb.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Set-Location $PSScriptRoot

Write-Host "=== 个人知识库启动 ===" -ForegroundColor Cyan
Write-Host "工作目录: $PWD"
if (-not (Test-Path ".env")) {
    Write-Host "[错误] 缺少 .env 文件（LLM/Embedding 配置）。请参考 env.example 配置。" -ForegroundColor Red
    exit 1
}

# 端口占用检查
$probe = New-Object System.Net.Sockets.TcpClient
try {
    $probe.Connect("127.0.0.1", 9621)
    $probe.Close()
    Write-Host "[提示] 端口 9621 已有服务在运行，直接访问 http://localhost:9621/console/" -ForegroundColor Yellow
    exit 0
} catch {
    $probe.Close()
}

Write-Host "启动 Docker 知识库: http://localhost:9621/console/"
docker compose up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "产品壳: http://localhost:9621/console/   API 文档: http://localhost:9621/docs"
Write-Host "原文目录: .\data\originals  索引目录: .\data\rag_storage"
