@echo off
chcp 65001 >nul
rem 一键启动个人知识库（等价于 start_kb.ps1）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_kb.ps1"
