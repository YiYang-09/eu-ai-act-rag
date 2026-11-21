# Run the full stack on Windows without Docker: GPU embedder/reranker, Qdrant in local file mode,
# LLM from the Ollama already installed on the machine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1            # set up (first time) and start
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -SkipSetup # start only
#
# API: http://localhost:8000/docs   UI: http://localhost:8501
param([switch]$SkipSetup, [string]$Model = "qwen2.5:7b")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $SkipSetup) {
    if (-not (Test-Path .venv)) { python -m venv .venv }
    .\.venv\Scripts\python -m pip install -q --upgrade pip
    # CUDA build of torch so bge-m3 and the reranker use the GPU
    .\.venv\Scripts\python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
    .\.venv\Scripts\python -m pip install -q -e ".[dense,serve,ui]"
    .\.venv\Scripts\python scripts\build_corpus.py --chunk-only
    # embeds on the GPU; re-running only embeds chunks whose text changed
    .\.venv\Scripts\python scripts\ingest.py --qdrant-path data\qdrant --device cuda
    ollama pull $Model
}

$env:LEXRAG_STORE = "qdrant"
$env:QDRANT_PATH = "data\qdrant"
$env:EMBED_DEVICE = "cuda"
$env:LLM_MODEL = $Model
$env:OLLAMA_URL = "http://localhost:11434"
$env:API_URL = "http://localhost:8000"

$api = Start-Process -PassThru -NoNewWindow .\.venv\Scripts\python -ArgumentList "-m","uvicorn","lexrag.serve.api:app","--port","8000"
Write-Host "API starting (loads models, ~30 s): http://localhost:8000/docs"
try {
    .\.venv\Scripts\python -m streamlit run app\streamlit_app.py --server.port 8501
} finally {
    Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
}
