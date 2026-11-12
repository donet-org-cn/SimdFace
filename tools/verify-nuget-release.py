"""校验 NuGet.org 提供的包与 CI 产物包含相同的文件内容。"""

import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def contents(package):
    """计算包内各文件的哈希，忽略 NuGet.org 添加的仓库签名。"""
    with zipfile.ZipFile(package) as archive:
        return {entry.filename: hashlib.sha256(archive.read(entry)).digest()
                for entry in archive.infolist() if entry.filename != ".signature.p7s"}


def verify(directory, version):
    """等待指定版本可下载，然后逐文件校验。"""
    filename = f"SimdFace.{version}.nupkg"
    expected = contents(directory / filename)
    url = f"https://api.nuget.org/v3-flatcontainer/simdface/{version}/{filename.lower()}"

    deadline = time.monotonic() + 1200
    while True:
        try:
            request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
            with urllib.request.urlopen(request, timeout=20) as response:
                package = response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (404, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        else:
            if contents(io.BytesIO(package)) != expected:
                raise SystemExit(f"NuGet.org 上的包内容与 CI 产物不同：{url}")
            print(f"已验证 {url}（SHA-256 {hashlib.sha256(package).hexdigest()}）", flush=True)
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(f"等待 NuGet.org 提供下载超时：{url}")
        time.sleep(min(20, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    arguments = parser.parse_args()
    verify(arguments.directory, arguments.version)
