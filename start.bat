@echo off
set DEV_API_KEY=dev-local-123
set CLAUDE_MODEL=claude-sonnet-4-5
cd /d %~dp0
py -3.11 -m uvicorn main:app --host 127.0.0.1 --port 8000
