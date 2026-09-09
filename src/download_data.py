from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PATH = DATA_DIR / "telco_customer_churn.csv"
URL = "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd4.8/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        print(f"Dataset already exists: {DATA_PATH}")
        return DATA_PATH
    print("Downloading IBM Telco Customer Churn dataset...")
    with urlopen(URL, timeout=30) as response:
        DATA_PATH.write_bytes(response.read())
    print(f"Saved {DATA_PATH}")
    return DATA_PATH


if __name__ == "__main__":
    download()
