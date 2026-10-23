param(
    [ValidateRange(1, 100000)]
    [int]$Limit = 100,
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
    & $uvExecutable --cache-dir '.\data\uv-cache' sync --upgrade --extra research
    if ($LASTEXITCODE -ne 0) { throw 'Dependency update failed. PyPI must be reachable from this terminal.' }
    & $uvExecutable run --no-sync python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed. No dataset API calls have started.' }
    & $uvExecutable run --no-sync python -m scripts.check_runtime
    if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed. Check the report in data/benchmarks/runtime-check.json.' }
    & $uvExecutable run --no-sync python -m scripts.check_gemini_model
    if ($LASTEXITCODE -ne 0) { throw 'Gemini model check failed. See data/benchmarks/gemini-model-check.json.' }
    $inputPath = '.\data\datasets\djinni\sample-100000.jsonl'
    if (-not (Test-Path -LiteralPath $inputPath)) {
        & $uvExecutable run --no-sync python -m scripts.prepare_djinni
        if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
    }
    & $uvExecutable run --no-sync python -m scripts.run_dataset --input $inputPath --limit $Limit --query $Query
    if ($LASTEXITCODE -ne 0) { throw 'Dataset run stopped. Saved ready CVs are reused on the next run.' }
} finally {
    Pop-Location
}
