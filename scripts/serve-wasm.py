"""Serve a published SimdFace WASM directory on localhost."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class WasmHandler(SimpleHTTPRequestHandler):
    extensions_map = SimpleHTTPRequestHandler.extensions_map | {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
    }


def main():
    default_dir = Path(__file__).resolve().parents[1] / "src/SimdFace.Browser/bin/Release/net10.0/publish/wwwroot"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=default_dir, type=Path)
    parser.add_argument("--port", type=int, default=5089)
    args = parser.parse_args()
    has_runtime = ((args.directory / "browser/deployment.json").is_file()
                   or (args.directory / "_framework").is_dir())
    if not (args.directory / "index.html").is_file() or not has_runtime:
        parser.error("Publish the browser project and pass its static output directory")
    handler = partial(WasmHandler, directory=str(args.directory.resolve()))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"SimdFace WASM is available at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
