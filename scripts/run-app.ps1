param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$SyncDependencies
)

$ErrorActionPreference = 'Stop'
$repositoryPath = Split-Path -Parent $PSScriptRoot
$appUrl = "http://127.0.0.1:$Port"
$serverProcess = $null

function Find-Executable {
    param([string]$Name, [string[]]$Candidates)
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    return $null
}

function Get-Json {
    param([string]$Url)
    try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 -ErrorAction Stop }
    catch { return $null }
}

function Test-App {
    $health = Get-Json "$appUrl/health"
    if (-not $health -or $health.status -ne 'ok') { return $false }
    $schema = Get-Json "$appUrl/openapi.json"
    return ($schema -and $schema.info.title -eq 'JD Matching')
}

function Test-Port {
    $client = New-Object System.Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $Port); return $true }
    catch { return $false }
    finally { $client.Dispose() }
}

Push-Location -LiteralPath $repositoryPath
try {
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
        Write-Warning 'Created .env. Set the API keys there to enable uploads and searches.'
    }

    $python = Join-Path $repositoryPath '.venv\Scripts\python.exe'
    $environmentReady = $false
    if ((Test-Path -LiteralPath $python) -and -not $SyncDependencies) {
        # Windows PowerShell 5 treats redirected native stderr as an error record.
        # A missing import is expected here: sync dependencies instead of aborting.
        $previousErrorPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $python -c 'import uvicorn, fastapi, chromadb, dotenv, multipart, pypdf, httpx, pyarrow; from langgraph.graph import StateGraph; from langgraph.checkpoint.sqlite import SqliteSaver; from google import genai; import langchain_google_genai' 2>$null | Out-Null
            $environmentReady = ($LASTEXITCODE -eq 0)
        } finally { $ErrorActionPreference = $previousErrorPreference }
    }
    if (-not $environmentReady) {
        $uv = Find-Executable 'uv' @(
            (Join-Path $repositoryPath 'data\tools\uv.exe'),
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe')
        )
        if (-not $uv) { throw 'uv was not found. Install uv, then run this script again: https://docs.astral.sh/uv/getting-started/installation/' }
        Write-Host 'Preparing Python and dependencies (including Corpus tools)...'
        & $uv --cache-dir '.\data\uv-cache' sync --locked --extra research --link-mode copy
        if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed. Check the output above and your Internet connection.' }
    }

    # Let python-dotenv and the app resolve settings, with the same precedence as
    # normal startup. Only non-secret settings and key-presence flags leave Python.
    $configProbe = @'
import json, os
from src.backend import config as c
required = set()
for provider in (c.CHAT_PROVIDER, c.EMBEDDING_PROVIDER):
    if provider == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        required.add("GEMINI_API_KEY")
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        required.add("OPENAI_API_KEY")
    elif provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        required.add("DEEPSEEK_API_KEY")
print(json.dumps({"ollama_url": c.OLLAMA_BASE_URL, "ollama_model": c.OLLAMA_CHAT_MODEL, "missing_keys": sorted(required)}))
'@
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $configOutput = $configProbe | & $python -
    } finally { $ErrorActionPreference = $previousErrorPreference }
    if ($LASTEXITCODE -ne 0) { throw 'Invalid app configuration. Check .env and the error above.' }
    $settings = $configOutput | ConvertFrom-Json
    foreach ($key in $settings.missing_keys) {
        Write-Warning "Missing $key in .env. The interface can open, but uploads/searches require the configured providers."
    }

    $logsPath = Join-Path $repositoryPath 'data\logs'
    New-Item -ItemType Directory -Path $logsPath -Force | Out-Null
    $runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $ollamaUrl = $settings.ollama_url.TrimEnd('/')
    $ollamaUri = [Uri]$ollamaUrl
    $ollama = Find-Executable 'ollama' @((Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'))
    $tags = Get-Json "$ollamaUrl/api/tags"
    if (-not $tags) {
        if (-not $ollamaUri.IsLoopback) { throw 'The configured remote Ollama server is unavailable. Check OLLAMA_BASE_URL in .env.' }
        if (-not $ollama) { throw 'Ollama was not found. Install it from https://ollama.com/download/windows and run this script again.' }
        Write-Host 'Starting Ollama...'
        $previousOllamaHost = $env:OLLAMA_HOST
        try {
            $env:OLLAMA_HOST = $ollamaUri.GetLeftPart([System.UriPartial]::Authority)
            $ollamaProcess = Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $logsPath "ollama-$runId.log") `
                -RedirectStandardError (Join-Path $logsPath "ollama-$runId.error.log")
        } finally { $env:OLLAMA_HOST = $previousOllamaHost }
        $deadline = (Get-Date).AddSeconds(30)
        do {
            $tags = Get-Json "$ollamaUrl/api/tags"
            if ($tags) { break }
            if ($ollamaProcess.HasExited) { break }
            Start-Sleep -Milliseconds 500
        } while ((Get-Date) -lt $deadline)
        if (-not $tags) { throw "Ollama did not become ready. See $logsPath\ollama-$runId.error.log" }
    }
    if (@($tags.models | ForEach-Object { $_.name }) -notcontains $settings.ollama_model) {
        if (-not $ollama) { throw "Model $($settings.ollama_model) is missing. Pull it on the configured Ollama server first." }
        Write-Host "Downloading Ollama model $($settings.ollama_model)..."
        $previousOllamaHost = $env:OLLAMA_HOST
        try {
            $env:OLLAMA_HOST = $ollamaUri.GetLeftPart([System.UriPartial]::Authority)
            & $ollama pull $settings.ollama_model
            if ($LASTEXITCODE -ne 0) { throw 'Ollama model download failed. Check your connection and retry.' }
        } finally { $env:OLLAMA_HOST = $previousOllamaHost }
    }
    Write-Host "Ollama ready: $($settings.ollama_model)"

    if (Test-App) {
        Write-Host "JD Matching is already running: $appUrl"
        if (-not $NoBrowser) { Start-Process $appUrl }
        exit 0
    }
    if (Test-Port) { throw "Port $Port is in use by another service. Try: .\run.bat -Port 8001" }

    $stdoutLog = Join-Path $logsPath "app-$runId.log"
    $stderrLog = Join-Path $logsPath "app-$runId.error.log"
    Write-Host "Starting JD Matching at $appUrl ..."
    $serverProcess = Start-Process -FilePath $python `
        -ArgumentList @('-m', 'uvicorn', 'src.backend.main:app', '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $repositoryPath -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    $deadline = (Get-Date).AddSeconds(60)
    $ready = $false
    do {
        if ($serverProcess.HasExited) { break }
        $ready = Test-App
        if ($ready) { break }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    if (-not $ready -or $serverProcess.HasExited) {
        if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 20 | Write-Host }
        throw "App startup failed. See $stderrLog"
    }
    Write-Host "Ready: $appUrl"
    Write-Host "API docs: $appUrl/docs"
    Write-Host "Logs: $stderrLog"
    Write-Host 'Keep this window open. Press Ctrl+C to stop the app. Ollama stays available.'
    if (-not $NoBrowser) { Start-Process $appUrl }
    while (-not $serverProcess.HasExited) { Start-Sleep -Seconds 1 }
    if ($serverProcess.ExitCode -ne 0) { throw "App exited unexpectedly. See $stderrLog" }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # Never stop a server that was already running before this script.
    if ($serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
