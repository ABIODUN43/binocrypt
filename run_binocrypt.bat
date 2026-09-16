@echo off
echo ===================================================
echo   Starting Binocrypt Quantitative Terminal
echo ===================================================
echo.
echo Starting FastAPI Backend on http://127.0.0.1:8001 ...
start "Binocrypt Backend" cmd /k "cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload"

echo Starting Vite Frontend on http://localhost:5174 ...
start "Binocrypt Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Binocrypt is launching! Open http://localhost:5174 in your browser.
echo ===================================================
pause
