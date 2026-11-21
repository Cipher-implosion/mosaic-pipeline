<#
  setup_windows.ps1 - one-time setup of the mosaic pipeline on Windows.

  Run from the project folder (the folder that contains mosaic.py):
      powershell -ExecutionPolicy Bypass -File setup_windows.ps1

  Options:
      -Cpu                force the CPU build of PyTorch even if an NVIDIA GPU is present
      -CudaIndex cu126    force a specific PyTorch CUDA wheel index (cu124, cu126, cu130, cu132, ...)

  Requirements: Python 3.11 or 3.12 (python.org, "Add python.exe to PATH"), Git.
  ffmpeg is needed to run the pipeline (winget install Gyan.FFmpeg); the script checks it.
  Safe to re-run: existing .venv, third_party\sam2 and model files are reused.
#>
param(
    [switch]$Cpu,
    [string]$CudaIndex = ""
)

# "Continue": every native command (pip, git, ...) is checked via $LASTEXITCODE; "Stop" would make
# Windows PowerShell 5.1 abort on harmless stderr output of native tools.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

function Step($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path (Join-Path $Root "mosaic.py"))) {
    Fail "mosaic.py not found in $Root. Run this script from the project folder."
}

# --------------------------------------------------------------------- Python
Step "Looking for Python 3.11 / 3.12"
$PyCmd = $null
foreach ($cand in @("py -3.12", "py -3.11", "python", "python3")) {
    $c = @($cand -split " ")
    $exe = $c[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $pyArgs = @($c | Select-Object -Skip 1)
    try {
        $ver = & $exe @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    } catch { continue }
    if ($LASTEXITCODE -eq 0 -and ($ver -eq "3.11" -or $ver -eq "3.12")) {
        $PyCmd = $c
        Ok "Python $ver  ($($c -join ' '))"
        break
    }
}
if (-not $PyCmd) {
    Fail ("Python 3.11 or 3.12 not found. Install it from https://www.python.org/downloads/windows/ " +
          "and tick 'Add python.exe to PATH' in the installer, then open a NEW PowerShell window.")
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "Git not found. Install it from https://git-scm.com/download/win (or: winget install Git.Git), then open a NEW PowerShell window."
}
Ok "Git found"

# ----------------------------------------------------------------------- venv
Step "Creating virtual environment .venv"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    $exe = $PyCmd[0]; $pyArgs = @($PyCmd | Select-Object -Skip 1)
    & $exe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "could not create .venv" }
    Ok "created .venv"
} else {
    Ok ".venv already exists, reusing it"
}
. (Join-Path $Root ".venv\Scripts\Activate.ps1")
Ok "activated .venv"

# pip output goes straight to the console; the function itself returns only True/False
function Pip { & $VenvPy -m pip @args | Out-Host; return ($LASTEXITCODE -eq 0) }

if (-not (Pip install --upgrade pip setuptools wheel)) { Fail "pip upgrade failed (check the internet connection)" }
Ok "pip upgraded"

# ---------------------------------------------------------------------- torch
Step "Installing PyTorch"
function CudaWorks {
    $r = & $VenvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
    return ($LASTEXITCODE -eq 0 -and "$r".Trim() -eq "True")
}

$HasGpu = $false
$DriverCuda = $null
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $smi = (& nvidia-smi 2>$null | Out-String)
    if ($LASTEXITCODE -eq 0) {
        $HasGpu = $true
        if ($smi -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
            $DriverCuda = [double]("{0}.{1}" -f $Matches[1], $Matches[2])
        }
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        Ok "NVIDIA GPU detected: $gpuName (driver supports CUDA $DriverCuda)"
    }
}
if (-not $HasGpu -and -not $Cpu) { Warn "nvidia-smi not found or failed -> no NVIDIA GPU, installing the CPU build" }

$TorchOk = $false
if ($HasGpu) {
    # Newest first. 'min' = minimum CUDA version the NVIDIA driver must report in nvidia-smi.
    $indexes = @(
        @{ tag = "cu132"; min = 13.2 },
        @{ tag = "cu130"; min = 13.0 },
        @{ tag = "cu126"; min = 12.6 },
        @{ tag = "cu124"; min = 12.4 }
    )
    if ($CudaIndex) {
        $tryList = @($CudaIndex)
    } elseif ($DriverCuda) {
        $tryList = @($indexes | Where-Object { $_.min -le $DriverCuda } | ForEach-Object { $_.tag })
    } else {
        $tryList = @($indexes | ForEach-Object { $_.tag })
    }
    if ($tryList.Count -eq 0) {
        Warn "The NVIDIA driver is too old for current PyTorch CUDA builds (needs CUDA 12.4+, has $DriverCuda)."
        Warn "Update the driver from https://www.nvidia.com/Download/index.aspx and re-run. Using CPU for now."
    }
    $installedOnce = $false
    foreach ($tag in $tryList) {
        Write-Host "  trying PyTorch wheel index $tag ..."
        $extra = @()
        if ($installedOnce) { $extra = @("--force-reinstall") }
        if (Pip install --upgrade @extra torch torchvision --index-url "https://download.pytorch.org/whl/$tag") {
            $installedOnce = $true
            if (CudaWorks) { $TorchOk = $true; Ok "PyTorch ($tag) installed, CUDA works"; break }
            Warn "PyTorch ($tag) installed but torch.cuda.is_available() is False; trying the next index"
        } else {
            Warn "install from $tag failed; trying the next index"
        }
    }
    if (-not $TorchOk) {
        Warn "No CUDA build of PyTorch worked on this PC -> installing the CPU build instead."
        if (-not (Pip install --upgrade --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/cpu")) {
            Fail "PyTorch (CPU) install failed"
        }
    }
} else {
    if (-not (Pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/cpu")) {
        Fail "PyTorch (CPU) install failed"
    }
    Ok "PyTorch (CPU) installed"
}

# ------------------------------------------------------- ultralytics + opencv
Step "Installing ultralytics and opencv-python"
if (-not (Pip install ultralytics opencv-python)) { Fail "ultralytics / opencv-python install failed" }
Ok "ultralytics + opencv-python installed"

# ----------------------------------------------------------------------- SAM 2
Step "Installing SAM 2 (third_party\sam2)"
$Sam2Dir = Join-Path $Root "third_party\sam2"
if (-not (Test-Path (Join-Path $Sam2Dir "setup.py"))) {
    New-Item -ItemType Directory -Force (Join-Path $Root "third_party") | Out-Null
    & git clone --depth 1 https://github.com/facebookresearch/sam2 $Sam2Dir
    if ($LASTEXITCODE -ne 0) { Fail "git clone of https://github.com/facebookresearch/sam2 failed" }
    Ok "cloned sam2"
} else {
    Ok "third_party\sam2 already present, skipping clone"
}

# --no-build-isolation: build against the torch installed above (otherwise pip pulls a
# separate CPU torch from PyPI into the build environment).
$sam2Installed = $false
if ($TorchOk) {
    Write-Host "  building with the optional CUDA extension (needs CUDA Toolkit + Visual Studio C++; may fail, that is OK)..."
    $env:SAM2_BUILD_CUDA = "1"; $env:SAM2_BUILD_ALLOW_ERRORS = "0"
    if (Pip install --no-build-isolation -e $Sam2Dir) {
        $sam2Installed = $true; Ok "SAM 2 installed with CUDA extension"
    } else {
        Warn "CUDA extension build failed -> retrying with SAM2_BUILD_CUDA=0 (the extension is optional)"
    }
}
if (-not $sam2Installed) {
    $env:SAM2_BUILD_CUDA = "0"; $env:SAM2_BUILD_ALLOW_ERRORS = "1"
    if (-not (Pip install --no-build-isolation -e $Sam2Dir)) { Fail "SAM 2 install failed" }
    Ok "SAM 2 installed (without the optional CUDA extension)"
}
Remove-Item Env:SAM2_BUILD_CUDA, Env:SAM2_BUILD_ALLOW_ERRORS -ErrorAction SilentlyContinue

# ---------------------------------------------------------------------- models
Step "Downloading models into models\"
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force $Models | Out-Null

function Get-Model($url, $name, $expectBytes) {
    $dst = Join-Path $Models $name
    if ((Test-Path $dst) -and ((Get-Item $dst).Length -gt 0) -and
        ($expectBytes -le 0 -or (Get-Item $dst).Length -eq $expectBytes)) {
        Ok "$name already present"; return
    }
    Write-Host "  downloading $name ..."
    $tmp = "$dst.part"
    try {
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    } catch {
        Remove-Item $tmp -ErrorAction SilentlyContinue
        Fail "download failed: $url ($($_.Exception.Message))"
    }
    $len = (Get-Item $tmp).Length
    if ($expectBytes -gt 0 -and $len -ne $expectBytes) {
        Warn "$name has $len bytes, expected $expectBytes (incomplete download?). Re-run the script if mosaic.py fails to load it."
    }
    Move-Item -Force $tmp $dst
    Ok "$name ($([math]::Round($len / 1MB, 1)) MB)"
}

Get-Model "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" "sam2.1_hiera_tiny.pt" 156008466
Get-Model "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" "yolo11n.pt" 5613764

# mosaic.py loads the config by its package name configs/sam2.1/sam2.1_hiera_t.yaml (inside
# the installed sam2); a copy is kept in models\ for reference, taken from the same checkout.
$cfgSrc = Join-Path $Sam2Dir "sam2\configs\sam2.1\sam2.1_hiera_t.yaml"
$cfgDst = Join-Path $Models "sam2.1_hiera_t.yaml"
if (Test-Path $cfgSrc) {
    Copy-Item -Force $cfgSrc $cfgDst; Ok "sam2.1_hiera_t.yaml (from third_party\sam2)"
} else {
    Get-Model "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml" "sam2.1_hiera_t.yaml" 0
}

# ---------------------------------------------------------------------- ffmpeg
Step "Checking ffmpeg"
$FfmpegOk = $false
if ((Get-Command ffmpeg -ErrorAction SilentlyContinue) -and (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    $v = (& ffmpeg -version 2>$null | Select-Object -First 1)
    if ($v) { $FfmpegOk = $true; Ok "$v" }
}
if (-not $FfmpegOk) {
    Warn "ffmpeg / ffprobe NOT found. mosaic.py needs them. Install with:"
    Write-Host "        winget install Gyan.FFmpeg" -ForegroundColor White
    Warn "then close this window, open a NEW PowerShell window (so PATH is refreshed) and check: ffmpeg -version"
}

# ---------------------------------------------------------------- final check
Step "Checking the Python environment"
& $VenvPy -c "import torch, torchvision, cv2, ultralytics, sam2; from sam2.build_sam import build_sam2_video_predictor; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
if ($LASTEXITCODE -ne 0) { Fail "import check failed (see the error above)" }
New-Item -ItemType Directory -Force (Join-Path $Root "test") | Out-Null

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Setup finished." -ForegroundColor Green
if (-not $FfmpegOk) { Write-Host " Remember: install ffmpeg first (winget install Gyan.FFmpeg)." -ForegroundColor Yellow }
Write-Host " Next steps (run in PowerShell, in this folder):"
Write-Host "   cd `"$Root`""
Write-Host ""
Write-Host " 1) GPU sanity check (prints True if the GPU will be used):"
Write-Host "   .\.venv\Scripts\python.exe -c `"import torch; print(torch.cuda.is_available())`""
Write-Host ""
Write-Host " 2) Make the synthetic test clip:"
Write-Host "   .\.venv\Scripts\python.exe make_test_clip.py test\test_clip.mp4"
Write-Host ""
Write-Host " 3) Run the pipeline on it:"
Write-Host "   .\.venv\Scripts\python.exe mosaic.py test\test_clip.mp4 test\out_test.mp4 --frame 0 --point `"96,180`" --detect-reacquire --detect-classes `"sports ball`" --save-masks test\out_test.masks.npz"
Write-Host "==================================================================" -ForegroundColor Cyan
