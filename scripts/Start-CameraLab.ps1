[CmdletBinding()]
param(
    [Parameter(Mandatory)][ipaddress] $LanAddress,
    [ValidateRange(1, 65535)][int] $HttpsPort = 5443,
    [ValidateRange(1, 65535)][int] $HttpPort = 5088
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$output = Join-Path $repo 'obj/camera-lan'
$publish = Join-Path $output 'app'
$certificates = Join-Path $output 'certificates'
$models = Join-Path $repo 'models'
$project = Join-Path $repo 'src/SimdFace.Web/SimdFace.Web.csproj'
$dotnet = (Get-Command dotnet -CommandType Application).Source
$listen = @(Get-NetTCPConnection -State Listen -LocalPort $HttpPort,$HttpsPort -ErrorAction SilentlyContinue |
    Where-Object OwningProcess -gt 0)
if ($listen.Count) {
    $ports = ($listen.LocalPort | Sort-Object -Unique) -join ', '
    throw "SimdFace ports are in use: $ports"
}

& $dotnet publish $project -c Release -o $publish --disable-build-servers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType Directory -Path $certificates -Force | Out-Null

$publicCert = Join-Path $certificates 'SimdFace-LAN.cer'
$privateCert = Join-Path $certificates 'server.pfx'
$protectedPassword = Join-Path $certificates 'server-password.dpapi'
$addressFile = Join-Path $certificates 'address.txt'
$reuse = (Test-Path -LiteralPath $publicCert) -and (Test-Path -LiteralPath $privateCert) -and
    (Test-Path -LiteralPath $protectedPassword) -and (Test-Path -LiteralPath $addressFile) -and
    ((Get-Content -LiteralPath $addressFile -Raw).Trim() -eq $LanAddress.ToString()) -and
    ((Get-Item -LiteralPath $privateCert).LastWriteTimeUtc -gt [DateTime]::UtcNow.AddDays(-80))

if ($reuse) {
    $securePassword = Get-Content -LiteralPath $protectedPassword -Raw | ConvertTo-SecureString
    $password = [Net.NetworkCredential]::new('', $securePassword).Password
}
else {
    $password = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(24))
    $caKey = [Security.Cryptography.RSA]::Create(3072)
    $serverKey = [Security.Cryptography.RSA]::Create(2048)
    try {
        $hash = [Security.Cryptography.HashAlgorithmName]::SHA256
        $padding = [Security.Cryptography.RSASignaturePadding]::Pkcs1
        $caRequest = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
            'CN=SimdFace LAN Test CA', $caKey, $hash, $padding)
        $caRequest.CertificateExtensions.Add(
            [Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($true, $true, 0, $true))
        $caRequest.CertificateExtensions.Add(
            [Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new('KeyCertSign, CrlSign', $true))
        $caRequest.CertificateExtensions.Add(
            [Security.Cryptography.X509Certificates.X509SubjectKeyIdentifierExtension]::new($caRequest.PublicKey, $false))

        $from = [DateTimeOffset]::UtcNow.AddMinutes(-10)
        $ca = $caRequest.CreateSelfSigned($from, $from.AddDays(100))
        try {
            $request = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
                "CN=$LanAddress", $serverKey, $hash, $padding)
            $request.CertificateExtensions.Add(
                [Security.Cryptography.X509Certificates.X509BasicConstraintsExtension]::new($false, $false, 0, $true))
            $request.CertificateExtensions.Add(
                [Security.Cryptography.X509Certificates.X509KeyUsageExtension]::new('DigitalSignature, KeyEncipherment', $true))
            $eku = [Security.Cryptography.OidCollection]::new()
            $eku.Add([Security.Cryptography.Oid]::new('1.3.6.1.5.5.7.3.1')) | Out-Null
            $request.CertificateExtensions.Add(
                [Security.Cryptography.X509Certificates.X509EnhancedKeyUsageExtension]::new($eku, $false))
            $names = [Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
            $names.AddIpAddress($LanAddress)
            $names.AddIpAddress([ipaddress]::Loopback)
            $names.AddDnsName('localhost')
            $request.CertificateExtensions.Add($names.Build())
            $serial = [Security.Cryptography.RandomNumberGenerator]::GetBytes(16)
            $serial[0] = $serial[0] -band 0x7f
            $issued = $request.Create($ca, $from, $from.AddDays(90), $serial)
            try {
                $leaf = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::CopyWithPrivateKey($issued, $serverKey)
                try {
                    [IO.File]::WriteAllBytes($privateCert, $leaf.Export('Pfx', $password))
                    [IO.File]::WriteAllBytes($publicCert, $ca.Export('Cert'))
                }
                finally { $leaf.Dispose() }
            }
            finally { $issued.Dispose() }
        }
        finally { $ca.Dispose() }
    }
    finally {
        $serverKey.Dispose()
        $caKey.Dispose()
    }
    ConvertTo-SecureString -String $password -AsPlainText -Force |
        ConvertFrom-SecureString | Set-Content -LiteralPath $protectedPassword -NoNewline
    Set-Content -LiteralPath $addressFile -Value $LanAddress.ToString() -NoNewline
}

Copy-Item -LiteralPath $publicCert -Destination (Join-Path $publish 'wwwroot/camera-ca.cer') -Force
$settings = @{
    Kestrel__Certificates__Default__Path = $privateCert
    Kestrel__Certificates__Default__Password = $password
    SimdFace__Models = $models
}
$dll = Join-Path $publish 'SimdFace.Web.dll'
$start = @{
    FilePath = $dotnet
    ArgumentList = ('"{0}" --urls "http://*:{1};https://*:{2}"' -f $dll, $HttpPort, $HttpsPort)
    WorkingDirectory = $publish
    Environment = $settings
    WindowStyle = 'Hidden'
    PassThru = $true
    RedirectStandardOutput = (Join-Path $output 'web.stdout.log')
    RedirectStandardError = (Join-Path $output 'web.stderr.log')
}
$process = Start-Process @start

for ($attempt = 0; $attempt -lt 25; $attempt++) {
    try {
        $ready = Invoke-WebRequest -Uri "http://127.0.0.1:$HttpPort/verify.html" -NoProxy -TimeoutSec 2
        if ($ready.StatusCode -eq 200) { break }
    }
    catch {
        if ($process.HasExited -or $attempt -eq 24) {
            Get-Content -LiteralPath $start.RedirectStandardError
            throw
        }
        Start-Sleep -Milliseconds 200
    }
}
Write-Output "SimdFace PID $($process.Id)"
Write-Output "Desktop camera: http://localhost:$HttpPort/verify.html"
Write-Output "Phone camera: https://$($LanAddress):$HttpsPort/verify.html"
Write-Output "Phone CA certificate: http://$($LanAddress):$HttpPort/camera-ca.cer"
Write-Output "CA SHA-256: $((Get-FileHash -LiteralPath $publicCert -Algorithm SHA256).Hash)"
