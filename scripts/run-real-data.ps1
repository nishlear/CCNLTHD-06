param(
    [ValidateRange(1, 100000)]
    [int]$Limit = 20,
    [string]$Query = "Python developer, at least 3 years, FastAPI"
)

$ErrorActionPreference = 'Stop'
$repositoryPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repositoryPath
try {
    if (-not (Test-Path -LiteralPath '.env')) {
        throw 'Configure GEMINI_API_KEY in .env before starting.'
    }
    $uvExecutable = Join-Path $repositoryPath 'data\tools\uv.exe'
    if (-not (Test-Path -LiteralPath $uvExecutable)) {
        $uvExecutable = (Get-Command uv -ErrorAction Stop).Source
    }
    & $uvExecutable --cache-dir '.\data\uv-cache' sync --locked --extra research --link-mode copy
    if ($LASTEXITCODE -ne 0) { throw 'Dependency update failed. PyPI must be reachable from this terminal.' }
    # Metadata alone can survive an incomplete package install. Verify imports
    # before pytest loads conftest, and repair the pinned package once if needed.
    $langgraphProbe = 'from langgraph.graph import END, START, StateGraph'
    & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -c $langgraphProbe
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'LangGraph installation is incomplete. Reinstalling the locked version.'
        & $uvExecutable --cache-dir '.\data\uv-cache' sync --locked --extra research --link-mode copy --reinstall-package langgraph
        if ($LASTEXITCODE -ne 0) { throw 'LangGraph repair failed. No dataset API calls have started.' }
        & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -c $langgraphProbe
        if ($LASTEXITCODE -ne 0) { throw 'LangGraph still cannot import after repair. No dataset API calls have started.' }
    }
    & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed. No dataset API calls have started.' }
    & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -m scripts.check_runtime
    if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed. Check the report in data/benchmarks/runtime-check.json.' }
    & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -m scripts.check_gemini_model --transport sdk
    if ($LASTEXITCODE -ne 0) { throw 'Gemini model check failed. See data/benchmarks/gemini-model-check.json.' }
    if ($Limit -gt 50) {
        Write-Warning 'Recommended -Limit per run is 20-50 (GEMINI_MAX_RPM=12 paces one Gemini call per 5 s).'
    }
    # The local fallback must be exactly qwen2.5:3b; no other Ollama model is used.
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Write-Warning 'Ollama not found: rate-limited extractions cannot fail over to qwen2.5:3b.'
    } elseif (-not ((& $ollama.Source list 2>$null) -match '^qwen2\.5:3b\s')) {
        Write-Warning 'qwen2.5:3b is not pulled. Run: ollama pull qwen2.5:3b'
    }
    $inputPath = '.\data\datasets\djinni\sample-100000.jsonl'
    if (-not (Test-Path -LiteralPath $inputPath)) {
        & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -m scripts.prepare_djinni
        if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
    }
    & $uvExecutable --cache-dir '.\data\uv-cache' run --no-sync python -m scripts.run_dataset --input $inputPath --limit $Limit --query $Query
    if ($LASTEXITCODE -ne 0) { throw 'Dataset run stopped. Saved ready CVs are reused on the next run.' }
} finally {
    Pop-Location
}
