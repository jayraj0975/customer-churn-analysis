"""Download IBM Telco Customer Churn dataset with robust error handling."""

import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PATH = DATA_DIR / "telco_customer_churn.csv"
URL = "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd4.8/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv"


def download():
    """
    Download IBM Telco Customer Churn dataset.

    Returns:
        Path: Path to the downloaded dataset.

    Raises:
        URLError: If download fails after retries.
        IOError: If unable to write file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print(f"✓ Dataset already exists: {DATA_PATH}")
        return DATA_PATH

    print("Downloading IBM Telco Customer Churn dataset...")
    print(f"  Source: {URL}")

    try:
        with urlopen(URL, timeout=30) as response:
            data = response.read()

        if not data:
            raise ValueError("Downloaded file is empty")

        DATA_PATH.write_bytes(data)
        file_size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Saved to {DATA_PATH} ({file_size_mb:.2f} MB)")
        return DATA_PATH

    except URLError as e:
        print(f"✗ Network error downloading dataset: {e}", file=sys.stderr)
        print(f"  Please check your internet connection and retry.", file=sys.stderr)
        print(f"  Or manually download from: {URL}", file=sys.stderr)
        raise

    except IOError as e:
        print(f"✗ Error writing to disk: {e}", file=sys.stderr)
        print(f"  Check that {DATA_DIR} is writable.", file=sys.stderr)
        raise

    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        download()
    except Exception as e:
        sys.exit(1)
