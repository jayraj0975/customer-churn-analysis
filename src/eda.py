"""Exploratory Data Analysis for customer churn dataset."""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_and_clean_data():
    """
    Load and clean the customer churn dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.

    Raises:
        FileNotFoundError: If dataset not found.
        ValueError: If dataset is malformed.
    """
    if not DATA.exists():
        print(f"✗ Dataset not found: {DATA}", file=sys.stderr)
        print(f"  Run: python src/download_data.py", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(DATA)
    except Exception as e:
        print(f"✗ Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert TotalCharges to numeric (handles strings with leading spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges
    initial_rows = len(df)
    df = df.dropna(subset=["TotalCharges"]).copy()
    dropped = initial_rows - len(df)

    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing TotalCharges")

    return df


def print_summary(df):
    """Print dataset summary statistics."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape:           {df.shape[0]} customers, {df.shape[1]} features")
    print(f"Churn rate:      {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"Churn cases:     {(df['Churn'] == 'Yes').sum()} customers")
    print(f"Non-churn cases: {(df['Churn'] == 'No').sum()} customers")

    missing = df.isna().sum()
    if missing.sum() > 0:
        print("\nMissing values (top 5):")
        for col, count in missing.sort_values(ascending=False).head().items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print("\nMissing values: None ✓")

    print("\n" + "=" * 70)


def plot_churn_by_categorical(df):
    """Generate bar plots of churn rate by categorical features."""
    print("\nGenerating categorical churn plots...")
    sns.set_theme(style="whitegrid")

    categorical_features = [
        ("Contract", "churn_by_contract.png"),
        ("InternetService", "churn_by_internet_service.png"),
        ("PaymentMethod", "churn_by_payment_method.png"),
        ("TechSupport", "churn_by_tech_support.png"),
        ("OnlineSecurity", "churn_by_online_security.png"),
    ]

    for col_name, filename in categorical_features:
        if col_name not in df.columns:
            print(f"  ⚠ Column '{col_name}' not found, skipping")
            continue

        rates = (
            df.groupby(col_name)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 5))
        rates.plot(kind="bar", color="steelblue")
        plt.ylabel("Churn Rate", fontsize=11)
        plt.title(f"Customer Churn Rate by {col_name}", fontsize=12, fontweight="bold")
        plt.xlabel(col_name, fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUT / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {filename}")


def plot_numeric_distributions(df):
    """Generate plots of numeric features by churn status."""
    print("\nGenerating numeric distribution plots...")

    # Tenure distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette=["green", "red"],
    )
    plt.title("Customer Tenure Distribution by Churn Status", fontsize=12, fontweight="bold")
    plt.xlabel("Tenure (months)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "tenure_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ tenure_by_churn.png")

    # Monthly charges distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="MonthlyCharges",
        hue="Churn",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette=["green", "red"],
    )
    plt.title("Monthly Charges Distribution by Churn Status", fontsize=12, fontweight="bold")
    plt.xlabel("Monthly Charges ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "monthly_charges_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ monthly_charges_by_churn.png")


def generate_business_summary(df):
    """Generate business-relevant summary statistics."""
    print("\nGenerating business summary...")

    # Summary by contract type
    summary = df.groupby("Contract").agg(
        customers=("customerID", "count"),
        churn_count=("Churn", lambda x: (x == "Yes").sum()),
        churn_rate=("Churn", lambda x: (x == "Yes").mean()),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_tenure=("tenure", "mean"),
    ).sort_values("churn_rate", ascending=False)

    summary.to_csv(ROOT / "reports" / "contract_summary.csv")
    print(f"  ✓ contract_summary.csv")

    # Display summary
    print("\n" + "=" * 70)
    print("BUSINESS INSIGHTS: CHURN BY CONTRACT TYPE")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)


def main():
    """Run full EDA pipeline."""
    print("Starting Exploratory Data Analysis...")
    print(f"Dataset path: {DATA}")

    df = load_and_clean_data()
    print_summary(df)
    plot_churn_by_categorical(df)
    plot_numeric_distributions(df)
    generate_business_summary(df)

    print("\n✓ EDA complete. Outputs saved to reports/")


if __name__ == "__main__":
    main()
