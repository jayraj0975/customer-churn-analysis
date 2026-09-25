"""Download the IBM Telco Customer Churn dataset and verify it.

The file is pinned by SHA-256, so a silently changed upstream file (or a
truncated download) is refused instead of quietly changing every result.
"""

import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256_of  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PATH = DATA_DIR / "telco_customer_churn.csv"
URL = "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd4.8/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv"
# Checked 2026-09-25: 7,043 data rows. If IBM ever changes the file this fails loudly;
# review the change, then update the hash on purpose.
EXPECTED_SHA256 = "3d5c233415c1b42bdea7172c73e620819f507f0a8294bc2337a1d8a8877feef0"
ATTEMPTS = 3


class ChecksumMismatch(RuntimeError):
    pass


def verify(path: Path = DATA_PATH) -> str:
    """Return the file's SHA-256, or raise ``ChecksumMismatch`` if it is not the pinned dataset."""
    digest = sha256_of(path)
    if digest != EXPECTED_SHA256:
        raise ChecksumMismatch(
            f"{path} has SHA-256 {digest}, expected {EXPECTED_SHA256}. "
            "Delete it and re-download, or update EXPECTED_SHA256 after reviewing the change."
        )
    return digest


def download() -> Path:
    """Fetch the dataset (retrying on network errors) and verify its checksum."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        verify(DATA_PATH)
        print(f"Dataset already present and verified: {DATA_PATH}")
        return DATA_PATH

    print(f"Downloading IBM Telco Customer Churn dataset from {URL}")
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urlopen(URL, timeout=30) as response:
                data = response.read()
            if not data:
                raise ValueError("downloaded file is empty")
            tmp = DATA_PATH.with_suffix(".part")
            tmp.write_bytes(data)
            try:
                verify(tmp)
            except ChecksumMismatch:
                tmp.unlink(missing_ok=True)
                raise
            tmp.replace(DATA_PATH)
            print(f"Saved and verified {DATA_PATH} ({len(data) / 1e6:.2f} MB)")
            return DATA_PATH
        except (URLError, TimeoutError, ValueError) as exc:
            last = exc
            print(f"Attempt {attempt}/{ATTEMPTS} failed: {exc}", file=sys.stderr)
            if attempt < ATTEMPTS:
                time.sleep(2 * attempt)
    raise RuntimeError(f"could not download the dataset: {last}. Source: {URL}") from last


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:  # noqa: BLE001 - CLI entry point: print and exit non-zero
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
