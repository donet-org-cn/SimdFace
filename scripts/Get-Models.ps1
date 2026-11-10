$ErrorActionPreference = 'Stop'
$modelDirectory = Join-Path $PSScriptRoot '..\models'
New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null

$baseUrl = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models'
$models = @(
    @{
        Name = 'face_detection_yunet_2023mar.onnx'
        Folder = 'face_detection_yunet'
        Sha256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
    }
    @{
        Name = 'face_recognition_sface_2021dec.onnx'
        Folder = 'face_recognition_sface'
        Sha256 = '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'
    }
)

foreach ($model in $models) {
    $destination = Join-Path $modelDirectory $model.Name
    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
        (Get-FileHash -LiteralPath $destination).Hash -eq $model.Sha256) {
        Write-Output "Verified $($model.Name)"
        continue
    }

    $partial = "$destination.$([Guid]::NewGuid().ToString('N')).partial"
    try {
        $url = "$baseUrl/$($model.Folder)/$($model.Name)"
        Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 720 -OutFile $partial
        if ((Get-FileHash -LiteralPath $partial).Hash -ne $model.Sha256) {
            throw "SHA-256 mismatch for $($model.Name)."
        }
        Move-Item -LiteralPath $partial -Destination $destination -Force
        Write-Output "Downloaded and verified $($model.Name)"
    }
    finally {
        if (Test-Path -LiteralPath $partial -PathType Leaf) {
            Remove-Item -LiteralPath $partial
        }
    }
}
