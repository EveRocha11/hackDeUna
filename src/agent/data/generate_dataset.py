"""Generate a reproducible synthetic DeUna-like transactional dataset.

This module creates CSV artifacts and a DuckDB file with four tables:
- merchants
- sellers
- customers
- transactions
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """Runtime configuration for dataset generation.

    Args:
        seed: Deterministic seed for all random operations.
        merchant_count: Number of merchants to generate.
        months: Number of months of transaction history.
        tx_per_merchant: Transaction count target per merchant.
        end_date: Final date included in generation window.
        timezone: IANA timezone name used for timestamps.
        currency: Currency code for all transactions.
        raw_dir: Output directory for CSV files.
        duckdb_path: Output path for DuckDB database file.

    Returns:
        DatasetConfig: Immutable config object.

    Raises:
        ValueError: If invalid numeric values are provided.
    """

    seed: int
    merchant_count: int
    months: int
    tx_per_merchant: int
    end_date: date
    timezone: str
    currency: str
    raw_dir: Path
    duckdb_path: Path

    def __post_init__(self) -> None:
        """Validate config values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If config constraints are violated.
        """
        if self.merchant_count < 1:
            raise ValueError("merchant_count must be >= 1")
        if self.months < 1:
            raise ValueError("months must be >= 1")
        if self.tx_per_merchant < 1:
            raise ValueError("tx_per_merchant must be >= 1")


@dataclass(frozen=True)
class MerchantAnomaly:
    """Defines deterministic anomaly periods per merchant.

    Args:
        merchant_id: Merchant identifier.
        start_date: Inclusive anomaly start date.
        end_date: Inclusive anomaly end date.
        multiplier: Demand multiplier for the anomaly period.

    Returns:
        MerchantAnomaly: Immutable anomaly definition.

    Raises:
        ValueError: If multiplier is not positive.
    """

    merchant_id: str
    start_date: date
    end_date: date
    multiplier: float

    def __post_init__(self) -> None:
        """Ensure anomaly multiplier is valid.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If multiplier is <= 0.
        """
        if self.multiplier <= 0:
            raise ValueError("multiplier must be > 0")


def _load_config() -> DatasetConfig:
    """Load dataset config from environment variables.

    Args:
        None.

    Returns:
        DatasetConfig: Parsed dataset configuration.

    Raises:
        ValueError: If DATASET_END_DATE format is invalid.
    """
    end_date = date.fromisoformat(os.getenv("DATASET_END_DATE", "2026-04-18"))
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    duckdb_path = Path(os.getenv("DATA_DUCKDB_PATH", "data/duckdb/analytics.duckdb"))

    return DatasetConfig(
        seed=int(os.getenv("DATASET_SEED", "20260418")),
        merchant_count=int(os.getenv("DATASET_MERCHANT_COUNT", "3")),
        months=int(os.getenv("DATASET_MONTHS", "12")),
        tx_per_merchant=int(os.getenv("DATASET_TX_PER_MERCHANT", "2000")),
        end_date=end_date,
        timezone=os.getenv("APP_TIMEZONE", "America/Guayaquil"),
        currency=os.getenv("APP_CURRENCY", "USD"),
        raw_dir=raw_dir,
        duckdb_path=duckdb_path,
    )


def _build_merchants(merchant_count: int) -> pd.DataFrame:
    """Create merchants with deterministic IDs and variety.

    Args:
        merchant_count: Number of merchants.

    Returns:
        pd.DataFrame: Merchant table.

    Raises:
        None.
    """
    categories = ["grocery", "food_stand", "beauty", "repair", "stationery"]
    cities = ["Quito", "Guayaquil", "Cuenca", "Manta", "Loja"]

    rows: list[dict[str, object]] = []
    for idx in range(merchant_count):
        merchant_id = f"M{idx + 1:03d}"
        rows.append(
            {
                "merchant_id": merchant_id,
                "merchant_name": f"Merchant {idx + 1}",
                "city": cities[idx % len(cities)],
                "category": categories[idx % len(categories)],
                "created_at": "2025-01-01",
            }
        )
    return pd.DataFrame(rows)


def _build_sellers(merchants: pd.DataFrame) -> pd.DataFrame:
    """Create owner and optional sellers per merchant.

    Args:
        merchants: Merchant table.

    Returns:
        pd.DataFrame: Sellers table with 1-3 sellers per merchant.

    Raises:
        None.
    """
    pattern = [1, 2, 3]
    rows: list[dict[str, object]] = []

    for idx, merchant in enumerate(merchants.itertuples(index=False), start=0):
        seller_count = pattern[idx % len(pattern)]
        merchant_id = str(merchant.merchant_id)
        for seller_idx in range(seller_count):
            role = "owner" if seller_idx == 0 else "seller"
            seller_suffix = "O" if seller_idx == 0 else f"S{seller_idx}"
            seller_id = f"{merchant_id}_{seller_suffix}"
            rows.append(
                {
                    "seller_id": seller_id,
                    "merchant_id": merchant_id,
                    "role": role,
                    "seller_display_name": f"{merchant_id} {role.title()} {seller_idx + 1}",
                }
            )

    return pd.DataFrame(rows)


def _build_customers(
    merchants: pd.DataFrame,
    rng: np.random.Generator,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Create per-merchant customer catalogs with frequency profiles.

    Args:
        merchants: Merchant table.
        rng: Numpy random generator.
        start_date: Earliest first_seen date.
        end_date: Latest first_seen date.

    Returns:
        pd.DataFrame: Customer catalog.

    Raises:
        None.
    """
    rows: list[dict[str, object]] = []
    profile_values = np.array(["loyal", "occasional", "one_time"], dtype=object)
    profile_probs = np.array([0.15, 0.55, 0.30], dtype=float)
    first_names = np.array(
        [
            "Ana",
            "Carlos",
            "Daniela",
            "Diego",
            "Elena",
            "Felipe",
            "Gabriela",
            "Javier",
            "Lucia",
            "Maria",
            "Miguel",
            "Paola",
            "Sofia",
            "Valeria",
        ],
        dtype=object,
    )
    last_names = np.array(
        [
            "Alvarado",
            "Cabrera",
            "Cevallos",
            "Gomez",
            "Guerrero",
            "Lopez",
            "Martinez",
            "Mendoza",
            "Moreira",
            "Paredes",
            "Quintero",
            "Rodriguez",
            "Sanchez",
            "Zambrano",
        ],
        dtype=object,
    )

    day_span = (end_date - start_date).days

    for _, merchant in merchants.iterrows():
        merchant_id = str(merchant["merchant_id"])
        customer_count = 450
        for idx in range(customer_count):
            customer_id = f"C_{merchant_id}_{idx + 1:04d}"
            profile = str(rng.choice(profile_values, p=profile_probs))
            first_seen = start_date + timedelta(days=int(rng.integers(0, max(day_span, 1))))
            customer_display_name = (
                f"{rng.choice(first_names)} {rng.choice(last_names)} {rng.choice(last_names)}"
            )

            doc = int(rng.integers(1_000_000_000, 9_999_999_999))
            acct = int(rng.integers(100_000_000_000, 999_999_999_999))
            rows.append(
                {
                    "customer_id": customer_id,
                    "merchant_id": merchant_id,
                    "customer_display_name": customer_display_name,
                    "frequency_profile": profile,
                    "first_seen_at": first_seen.isoformat(),
                    "id_doc_masked": f"***{str(doc)[-4:]}",
                    "account_masked": f"***{str(acct)[-4:]}",
                }
            )

    return pd.DataFrame(rows)


def _daily_weights(
    date_index: pd.DatetimeIndex,
    merchant_id: str,
    anomalies: list[MerchantAnomaly],
) -> np.ndarray:
    """Create weighted probabilities per day for transaction sampling.

    Args:
        date_index: Date range used for generation.
        merchant_id: Active merchant ID.
        anomalies: List of anomaly definitions.

    Returns:
        np.ndarray: Normalized daily probability weights.

    Raises:
        None.
    """
    weekday_multiplier = {
        0: 1.05,
        1: 1.00,
        2: 1.00,
        3: 1.02,
        4: 1.10,
        5: 1.18,
        6: 0.92,
    }

    values: list[float] = []
    for current_day in date_index:
        base = weekday_multiplier[current_day.weekday()]

        month_factor = 1.0 + ((current_day.month % 4) * 0.03)
        score = base * month_factor

        for anomaly in anomalies:
            if anomaly.merchant_id != merchant_id:
                continue
            if anomaly.start_date <= current_day.date() <= anomaly.end_date:
                score *= anomaly.multiplier

        values.append(score)

    weights = np.array(values, dtype=float)
    return weights / weights.sum()


def _amount_for_merchant(merchant_id: str, rng: np.random.Generator) -> float:
    """Sample transaction amount using merchant-specific scale.

    Args:
        merchant_id: Merchant identifier.
        rng: Numpy random generator.

    Returns:
        float: Rounded transaction amount in USD.

    Raises:
        None.
    """
    merchant_scale = {
        "M001": (2.3, 0.42),
        "M002": (2.5, 0.38),
        "M003": (2.4, 0.40),
    }
    mean, sigma = merchant_scale.get(merchant_id, (2.35, 0.4))
    amount = float(rng.lognormal(mean=mean, sigma=sigma))
    return round(max(1.0, min(amount, 250.0)), 2)


def _customer_weights(customers_for_merchant: pd.DataFrame) -> np.ndarray:
    """Build weighted probabilities by customer profile.

    Args:
        customers_for_merchant: Customer rows for one merchant.

    Returns:
        np.ndarray: Normalized customer selection probabilities.

    Raises:
        None.
    """
    profile_weight = {"loyal": 8.0, "occasional": 3.0, "one_time": 1.0}
    base = customers_for_merchant["frequency_profile"].map(profile_weight).astype(float).to_numpy()
    return base / base.sum()


def _seller_weights(sellers_for_merchant: pd.DataFrame) -> np.ndarray:
    """Create seller assignment probabilities.

    Args:
        sellers_for_merchant: Seller rows for one merchant.

    Returns:
        np.ndarray: Normalized seller probabilities.

    Raises:
        None.
    """
    count = len(sellers_for_merchant)
    if count == 1:
        return np.array([1.0], dtype=float)
    if count == 2:
        return np.array([0.60, 0.40], dtype=float)
    return np.array([0.50, 0.30, 0.20], dtype=float)


def _build_transactions(
    config: DatasetConfig,
    merchants: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    rng: np.random.Generator,
    anomalies: list[MerchantAnomaly],
) -> pd.DataFrame:
    """Generate transaction rows for all merchants.

    Args:
        config: Dataset configuration.
        merchants: Merchant table.
        sellers: Sellers table.
        customers: Customers table.
        rng: Numpy random generator.
        anomalies: Deterministic anomaly list.

    Returns:
        pd.DataFrame: Transactions table.

    Raises:
        None.
    """
    tz = ZoneInfo(config.timezone)
    start_date = config.end_date - timedelta(days=(config.months * 30 - 1))
    date_index = pd.date_range(start=start_date, end=config.end_date, freq="D")

    hours = np.array([7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20], dtype=int)
    hour_probs = np.array([0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.11, 0.08, 0.10, 0.11, 0.08, 0.06])

    rows: list[dict[str, object]] = []
    tx_global_counter = 0

    for _, merchant in merchants.iterrows():
        merchant_id = str(merchant["merchant_id"])
        day_weights = _daily_weights(date_index, merchant_id, anomalies)

        merchant_customers = customers[customers["merchant_id"] == merchant_id].reset_index(drop=True)
        cust_probs = _customer_weights(merchant_customers)

        merchant_sellers = sellers[sellers["merchant_id"] == merchant_id].reset_index(drop=True)
        seller_probs = _seller_weights(merchant_sellers)

        sampled_days = rng.choice(date_index.to_numpy(), size=config.tx_per_merchant, p=day_weights)

        for local_idx, sampled_day in enumerate(sampled_days):
            tx_global_counter += 1
            tx_id = f"T_{merchant_id}_{local_idx + 1:05d}"

            sampled_customer_idx = int(rng.choice(len(merchant_customers), p=cust_probs))
            customer_row = merchant_customers.iloc[sampled_customer_idx]

            sampled_seller_idx = int(rng.choice(len(merchant_sellers), p=seller_probs))
            seller_row = merchant_sellers.iloc[sampled_seller_idx]

            hour = int(rng.choice(hours, p=hour_probs))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))

            day_py = pd.Timestamp(sampled_day).date()
            occurred_at = datetime.combine(day_py, time(hour=hour, minute=minute, second=second), tz)

            rows.append(
                {
                    "transaction_id": tx_id,
                    "merchant_id": merchant_id,
                    "seller_id": str(seller_row["seller_id"]),
                    "customer_id": str(customer_row["customer_id"]),
                    "occurred_at": occurred_at.isoformat(),
                    "amount": _amount_for_merchant(merchant_id, rng),
                    "currency": config.currency,
                    "payer_doc_masked": str(customer_row["id_doc_masked"]),
                    "payer_account_masked": str(customer_row["account_masked"]),
                    "payment_channel": "deuna_quick_transfer",
                    "week_start_day": "MONDAY",
                    "tx_sequence": tx_global_counter,
                }
            )

    return pd.DataFrame(rows)


def _write_csvs(
    config: DatasetConfig,
    merchants: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """Persist generated dataframes as CSV files.

    Args:
        config: Dataset configuration.
        merchants: Merchant table.
        sellers: Seller table.
        customers: Customer table.
        transactions: Transaction table.

    Returns:
        None.

    Raises:
        OSError: If output paths cannot be created.
    """
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    merchants.to_csv(config.raw_dir / "merchants.csv", index=False)
    sellers.to_csv(config.raw_dir / "sellers.csv", index=False)
    customers.to_csv(config.raw_dir / "customers.csv", index=False)
    transactions.to_csv(config.raw_dir / "transactions.csv", index=False)


def _write_duckdb(
    config: DatasetConfig,
    merchants: pd.DataFrame,
    sellers: pd.DataFrame,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """Create DuckDB tables from generated dataframes.

    Args:
        config: Dataset configuration.
        merchants: Merchant table.
        sellers: Seller table.
        customers: Customer table.
        transactions: Transaction table.

    Returns:
        None.

    Raises:
        duckdb.Error: If SQL execution fails.
    """
    config.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(config.duckdb_path)) as conn:
        conn.register("merchants_df", merchants)
        conn.register("sellers_df", sellers)
        conn.register("customers_df", customers)
        conn.register("transactions_df", transactions)

        conn.execute("CREATE OR REPLACE TABLE merchants AS SELECT * FROM merchants_df")
        conn.execute("CREATE OR REPLACE TABLE sellers AS SELECT * FROM sellers_df")
        conn.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM customers_df")
        conn.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM transactions_df")


def _build_anomalies(end_date: date) -> list[MerchantAnomaly]:
    """Define deterministic anomaly windows for testable insights.

    Args:
        end_date: Last generation date.

    Returns:
        list[MerchantAnomaly]: Anomaly definitions.

    Raises:
        None.
    """
    return [
        MerchantAnomaly(
            merchant_id="M001",
            start_date=end_date - timedelta(days=49),
            end_date=end_date - timedelta(days=43),
            multiplier=0.45,
        ),
        MerchantAnomaly(
            merchant_id="M002",
            start_date=end_date - timedelta(days=84),
            end_date=end_date - timedelta(days=78),
            multiplier=1.60,
        ),
    ]


def _write_metadata(config: DatasetConfig, anomalies: list[MerchantAnomaly]) -> None:
    """Persist generation metadata for traceability.

    Args:
        config: Dataset configuration.
        anomalies: Injected anomaly definitions.

    Returns:
        None.

    Raises:
        OSError: If metadata file cannot be written.
    """
    payload = {
        "seed": config.seed,
        "merchant_count": config.merchant_count,
        "months": config.months,
        "tx_per_merchant": config.tx_per_merchant,
        "end_date": config.end_date.isoformat(),
        "timezone": config.timezone,
        "currency": config.currency,
        "week_definition": "monday_sunday",
        "anomalies": [
            {
                "merchant_id": anomaly.merchant_id,
                "start_date": anomaly.start_date.isoformat(),
                "end_date": anomaly.end_date.isoformat(),
                "multiplier": anomaly.multiplier,
            }
            for anomaly in anomalies
        ],
    }
    with (config.raw_dir / "generation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def generate_dataset(config: DatasetConfig) -> None:
    """Generate and persist all dataset artifacts.

    Args:
        config: Dataset configuration.

    Returns:
        None.

    Raises:
        ValueError: If config is invalid.
        duckdb.Error: If database write fails.
        OSError: If output write fails.
    """
    rng = np.random.default_rng(config.seed)
    merchants = _build_merchants(config.merchant_count)
    sellers = _build_sellers(merchants)

    start_date = config.end_date - timedelta(days=(config.months * 30 - 1))
    customers = _build_customers(merchants, rng, start_date, config.end_date)
    anomalies = _build_anomalies(config.end_date)
    transactions = _build_transactions(config, merchants, sellers, customers, rng, anomalies)

    _write_csvs(config, merchants, sellers, customers, transactions)
    _write_duckdb(config, merchants, sellers, customers, transactions)
    _write_metadata(config, anomalies)


def main() -> None:
    """CLI entrypoint for dataset generation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If argparse parsing fails.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic DeUna MVP dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override. If omitted, DATASET_SEED is used.",
    )
    args = parser.parse_args()

    config = _load_config()
    if args.seed is not None:
        config = DatasetConfig(
            seed=args.seed,
            merchant_count=config.merchant_count,
            months=config.months,
            tx_per_merchant=config.tx_per_merchant,
            end_date=config.end_date,
            timezone=config.timezone,
            currency=config.currency,
            raw_dir=config.raw_dir,
            duckdb_path=config.duckdb_path,
        )

    generate_dataset(config)
    print("Dataset generated successfully.")
    print(f"CSV output: {config.raw_dir}")
    print(f"DuckDB file: {config.duckdb_path}")


if __name__ == "__main__":
    main()
