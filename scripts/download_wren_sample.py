"""Cache WrenAI's original E-commerce Parquet files for the local data mode."""

import os
import re
import subprocess
import tempfile
from pathlib import Path


def is_parquet(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as stream:
        if stream.read(4) != b"PAR1":
            return False
        stream.seek(-4, 2)
        return stream.read(4) == b"PAR1"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    source = root / "WrenAI/wren-ui/src/apollo/server/data/sample.ts"
    urls = sorted(set(re.findall(
        r"https://assets\.getwren\.ai/sample_data/brazilian-ecommerce/[a-z_]+\.parquet",
        source.read_text(),
    )))
    if len(urls) != 9:
        raise RuntimeError("Expected nine original WrenAI E-commerce sample files")
    destination = Path(os.environ.get(
        "WREN_SAMPLE_DATA_DIR", str(root / ".local/wren-data/ecommerce")
    ))
    destination.mkdir(parents=True, exist_ok=True)
    for url in urls:
        target = destination / url.rsplit("/", 1)[1]
        if is_parquet(target):
            continue
        print(f"Downloading {target.name}...", flush=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
                temporary = Path(output.name)
            # macOS curl uses the system certificate store; no extra Python package.
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "2", "--connect-timeout", "15", "--max-time", "120",
                "--output", str(temporary), url,
            ], check=True)
            if not is_parquet(temporary):
                raise RuntimeError(f"Invalid Parquet download: {target.name}")
            temporary.replace(target)
            print(f"  {target.stat().st_size / (1024 * 1024):.1f} MiB", flush=True)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    print(f"E-commerce sample ready: {destination}", flush=True)


if __name__ == "__main__":
    main()
