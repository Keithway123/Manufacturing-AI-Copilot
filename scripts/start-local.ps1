$ErrorActionPreference = "Stop"

# 根据脚本位置找到项目，不依赖终端当前目录。
$projectDir = Split-Path -Parent $PSScriptRoot
$dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$appUrl = "http://127.0.0.1:8001"

function Test-DockerReady {
    try {
        docker info *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

Push-Location $projectDir

try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker command not found."
    }

    if (-not (Test-DockerReady)) {
        Write-Host "Starting Docker Desktop..."
        Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    }

    $deadline = (Get-Date).AddSeconds(120)

    while (-not (Test-DockerReady)) {
        if ((Get-Date) -ge $deadline) {
            throw "Docker did not become ready within 120 seconds."
        }

        Start-Sleep -Seconds 2
    }

    # 日常启动复用现有镜像，不重建索引或初始化数据。
    Write-Host "Starting Manufacturing AI Copilot..."
    docker compose up -d manufacturing-api

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start project services. Check the output above."
    }

    $apiReady = $false
    $deadline = (Get-Date).AddSeconds(60)

    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest `
                -Uri "$appUrl/health" `
                -UseBasicParsing `
                -TimeoutSec 3

            if ($response.StatusCode -eq 200) {
                $apiReady = $true
                break
            }
        }
        catch {
            # API 启动期间可能暂时无法连接，稍后重试。
        }

        Start-Sleep -Seconds 2
    }

    if (-not $apiReady) {
        throw "API not ready. Run: docker compose logs --tail 50 manufacturing-api"
    }

    Start-Process $appUrl
    Write-Host "Ready: $appUrl"
}
catch {
    Write-Host "Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to close"
}
finally {
    Pop-Location
}