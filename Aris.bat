@echo off
echo Starting Ollama...

:: Start Ollama in background
start cmd /k "ollama serve"

:: Wait a bit (important)
timeout /t 5

echo Starting Llama3...
start cmd /k "ollama run llama3"

:: Wait again
timeout /t 5

echo Starting Streamlit App...
start cmd /k "streamlit run app.py"

echo All services started 🚀
pause