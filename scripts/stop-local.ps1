$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot

Push-Location $projectDir

try {
    Write-Host "Stopping Manufacturing AI Copilot..."

    docker compose stop manufacturing-api postgres qdrant

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stop project services. Check the output above."
    }

    Write-Host "Stopped. Containers and data are preserved."
}
catch {
    Write-Host "Stop failed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to close"
}
finally {
    Pop-Location
}