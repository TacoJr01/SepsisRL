@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "DOCKER_BUILD_TIMEOUT=600"
set "PING_URL=%~1"
set "REPO_DIR=%~2"

if "%PING_URL%"=="" (
  echo Usage: %~nx0 ^<ping_url^> [repo_dir]
  echo.
  echo   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
  echo   repo_dir   Path to your repo (default: current directory)
  exit /b 1
)

if "%REPO_DIR%"=="" set "REPO_DIR=%CD%"
for %%I in ("%REPO_DIR%") do set "REPO_DIR=%%~fI"

if "%PING_URL:~-1%"=="/" set "PING_URL=%PING_URL:~0,-1%"

set "PASS=0"

echo.
echo ========================================
echo   OpenEnv Submission Validator (Windows)
echo ========================================
echo Repo:     %REPO_DIR%
echo Ping URL: %PING_URL%
echo.

echo Step 1/3: Pinging HF Space (%PING_URL%/reset) ...

set "HTTP_CODE=000"
for /f %%A in ('powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try { $resp=Invoke-WebRequest -Method Post -Uri '%PING_URL%/reset' -Headers @{ 'Content-Type'='application/json' } -Body '{}' -TimeoutSec 30; $code=$resp.StatusCode } catch { if ($_.Exception.Response -and $_.Exception.Response.StatusCode) { $code=$_.Exception.Response.StatusCode.value__ } else { $code=0 } }; Write-Output $code"') do set "HTTP_CODE=%%A"

if "%HTTP_CODE%"=="200" (
  echo PASSED -- HF Space is live and responds to /reset
) else if "%HTTP_CODE%"=="000" (
  echo FAILED -- HF Space not reachable (connection failed or timed out)
  echo   Hint: Check your network connection and that the Space is running.
  echo   Hint: Try: curl -s -o nul -w "%%{http_code}" -X POST %PING_URL%/reset
  exit /b 1
) else (
  echo FAILED -- HF Space /reset returned HTTP %HTTP_CODE% (expected 200)
  echo   Hint: Make sure your Space is running and the URL is correct.
  echo   Hint: Try opening %PING_URL% in your browser first.
  exit /b 1
)

echo Step 2/3: Running docker build ...

where docker >nul 2>nul
if errorlevel 1 (
  echo FAILED -- docker command not found
  echo   Hint: Install Docker: https://docs.docker.com/get-docker/
  exit /b 1
)

if exist "%REPO_DIR%\Dockerfile" (
  set "DOCKER_CONTEXT=%REPO_DIR%"
) else if exist "%REPO_DIR%\server\Dockerfile" (
  set "DOCKER_CONTEXT=%REPO_DIR%\server"
) else (
  echo FAILED -- No Dockerfile found in repo root or server directory
  exit /b 1
)

echo   Found Dockerfile in %DOCKER_CONTEXT%

powershell -NoProfile -Command "$ctx='%DOCKER_CONTEXT%'; $p=Start-Process -FilePath docker -ArgumentList @('build', $ctx) -NoNewWindow -PassThru; if ($p.WaitForExit(%DOCKER_BUILD_TIMEOUT%*1000)) { exit $p.ExitCode } else { try { $p.Kill() } catch {}; exit 124 }"
if errorlevel 1 (
  echo FAILED -- Docker build failed or timed out (%DOCKER_BUILD_TIMEOUT%s)
  exit /b 1
) else (
  echo PASSED -- Docker build succeeded
)

echo Step 3/3: Running openenv validate ...

where openenv >nul 2>nul
if errorlevel 1 (
  echo FAILED -- openenv command not found
  echo   Hint: Install it: pip install openenv-core
  exit /b 1
)

pushd "%REPO_DIR%"
openenv validate
set "VALIDATE_RC=%ERRORLEVEL%"
popd

if not "%VALIDATE_RC%"=="0" (
  echo FAILED -- openenv validate failed
  exit /b 1
) else (
  echo PASSED -- openenv validate passed
)

echo.
echo ========================================
echo   All 3/3 checks passed!
echo   Your submission is ready to submit.
echo ========================================
echo.

exit /b 0
