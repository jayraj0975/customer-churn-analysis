from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    print("Shape:", df.shape)
    print("Churn rate:", round((df["Churn"] == "Yes").mean(), 4))
    print("Missing values:\n", df.isna().sum().sort_values(ascending=False).head(10))

    sns.set_theme(style="whitegrid")
    for column, filename in [
        ("Contract", "churn_by_contract.png"),
        ("InternetService", "churn_by_internet_service.png"),
        ("PaymentMethod", "churn_by_payment_method.png"),
    ]:
        rates = df.groupby(column)["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
        plt.figure(figsize=(9, 5))
        rates.plot(kind="bar")
        plt.ylabel("Churn rate")
        plt.title(f"Churn rate by {column}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, element="step", stat="density", common_norm=False)
    plt.title("Tenure distribution by churn")
    plt.tight_layout()
    plt.savefig(OUT / "tenure_by_churn.png", dpi=150)
    plt.close()

    summary = df.groupby("Contract").agg(
        customers=("customerID", "count"),
        churn_rate=("Churn", lambda x: (x == "Yes").mean()),
        avg_monthly_charges=("MonthlyCharges", "mean"),
    ).sort_values("churn_rate", ascending=False)
    summary.to_csv(ROOT / "reports" / "contract_summary.csv")
    print("EDA outputs written to reports/.")


if __name__ == "__main__":
    main()
