"""Validate synthetic dataset integrity and basic business sanity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime config for dataset validation.

    Args:
        raw_dir: Directory with generated CSV files.
        duckdb_path: Path to generated DuckDB file.
        expected_currency: Expected currency code.
        expected_timezone_suffix: Expected timezone suffix for ISO timestamps.
        expected_merchants: Number of merchants expected in the dataset.
        expected_tx_per_merchant: Approximate transactions per merchant.

    Returns:
        ValidationConfig: Immutable config object.

    Raises:
        ValueError: If invalid values are provided.
    """

    raw_dir: Path
    duckdb_path: Path
    expected_currency: str
    expected_timezone_suffix: str
    expected_merchants: int
    expected_tx_per_merchant: int

    def __post_init__(self) -> None:
        """Validate config values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If numeric constraints are invalid.
        """
        if self.expected_merchants < 1:
            raise ValueError("expected_merchants must be >= 1")
        if self.expected_tx_per_merchant < 1:
            raise ValueError("expected_tx_per_merchant must be >= 1")


def _load_config() -> ValidationConfig:
    """Load validation config from environment variables.

    Args:
        None.

    Returns:
        ValidationConfig: Parsed validation configuration.

    Raises:
        None.
    """
    return ValidationConfig(
        raw_dir=Path(os.getenv("DATA_RAW_DIR", "data/raw")),
        duckdb_path=Path(os.getenv("DATA_DUCKDB_PATH", "data/duckdb/analytics.duckdb")),
        expected_currency=os.getenv("APP_CURRENCY", "USD"),
        expected_timezone_suffix="-05:00",
        expected_merchants=int(os.getenv("DATASET_MERCHANT_COUNT", "3")),
        expected_tx_per_merchant=int(os.getenv("DATASET_TX_PER_MERCHANT", "2000")),
    )


def _sha256_of_file(path: Path) -> str:
    """Compute SHA256 hash for a file.

    Args:
        path: File path.

    Returns:
        str: SHA256 hex digest.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hash(raw_dir: Path) -> str:
    """Compute deterministic combined dataset hash from core CSV files.

    Args:
        raw_dir: CSV directory path.

    Returns:
        str: Combined hash digest.

    Raises:
        OSError: If one of the CSV files cannot be read.
    """
    core_files = ["merchants.csv", "sellers.csv", "customers.csv", "transactions.csv"]
    digest = hashlib.sha256()
    for filename in core_files:
        digest.update(_sha256_of_file(raw_dir / filename).encode("utf-8"))
    return digest.hexdigest()


def _check_timestamp_suffix(raw_dir: Path, expected_suffix: str) -> tuple[bool, str]:
    """Check timestamp timezone suffix in transaction CSV.

    Args:
        raw_dir: CSV directory path.
        expected_suffix: Expected ISO timezone suffix.

    Returns:
        tuple[bool, str]: Validation pass flag and detail message.

    Raises:
        OSError: If transactions file cannot be read.
    """
    tx_file = raw_dir / "transactions.csv"
    with tx_file.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            ts = row["occurred_at"]
            if not ts.endswith(expected_suffix):
                return False, f"Row {idx + 2} has unexpected timezone suffix: {ts}"
            if idx > 500:
                break
    return True, "Timestamp suffix check passed on first 500 rows."


def _run_duckdb_checks(config: ValidationConfig) -> dict[str, object]:
    """Run SQL integrity and sanity checks in DuckDB.

    Args:
        config: Validation configuration.

    Returns:
        dict[str, object]: Raw check outputs for report building.

    Raises:
        duckdb.Error: If SQL execution fails.
    """
    results: dict[str, object] = {}
    with duckdb.connect(str(config.duckdb_path), read_only=True) as conn:
        merchants = conn.execute("SELECT COUNT(*) FROM merchants").fetchone()[0]
        sellers_missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM transactions t
            LEFT JOIN sellers s ON t.seller_id = s.seller_id AND t.merchant_id = s.merchant_id
            WHERE s.seller_id IS NULL
            """
        ).fetchone()[0]
        customers_missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM transactions t
            LEFT JOIN customers c ON t.customer_id = c.customer_id AND t.merchant_id = c.merchant_id
            WHERE c.customer_id IS NULL
            """
        ).fetchone()[0]
        invalid_amount = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE amount <= 0"
        ).fetchone()[0]
        invalid_currency = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE currency <> ?",
            [config.expected_currency],
        ).fetchone()[0]
        duplicate_tx = conn.execute(
            """
            SELECT COUNT(*)
            FROM (
              SELECT transaction_id, COUNT(*) AS c
              FROM transactions
              GROUP BY transaction_id
              HAVING COUNT(*) > 1
            ) x
            """
        ).fetchone()[0]
        tx_without_seller = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE seller_id IS NULL OR seller_id = ''"
        ).fetchone()[0]

        tx_count_by_merchant = conn.execute(
            """
            SELECT merchant_id, COUNT(*) AS tx_count
            FROM transactions
            GROUP BY merchant_id
            ORDER BY merchant_id
            """
        ).fetchall()

        top_customer_share = conn.execute(
            """
            WITH customer_income AS (
              SELECT merchant_id, customer_id, SUM(amount) AS income
              FROM transactions
              GROUP BY merchant_id, customer_id
            ), ranked AS (
              SELECT merchant_id, income,
                     ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY income DESC) AS rn,
                     SUM(income) OVER (PARTITION BY merchant_id) AS total_income
              FROM customer_income
            )
            SELECT merchant_id,
                   ROUND(MAX(CASE WHEN rn = 1 THEN income / NULLIF(total_income, 0) END), 4) AS top1_share
            FROM ranked
            GROUP BY merchant_id
            ORDER BY merchant_id
            """
        ).fetchall()

        results.update(
            {
                "merchant_count": merchants,
                "sellers_missing_refs": sellers_missing,
                "customers_missing_refs": customers_missing,
                "invalid_amount_rows": invalid_amount,
                "invalid_currency_rows": invalid_currency,
                "duplicate_transaction_ids": duplicate_tx,
                "transactions_without_seller": tx_without_seller,
                "tx_count_by_merchant": [
                    {"merchant_id": row[0], "tx_count": int(row[1])} for row in tx_count_by_merchant
                ],
                "top_customer_share": [
                    {"merchant_id": row[0], "top1_share": float(row[1])}
                    for row in top_customer_share
                ],
            }
        )
    return results


def validate_dataset(config: ValidationConfig) -> tuple[bool, dict[str, object]]:
    """Validate generated dataset and return a report payload.

    Args:
        config: Validation configuration.

    Returns:
        tuple[bool, dict[str, object]]: Overall pass flag and full report payload.

    Raises:
        OSError: If files are missing.
        duckdb.Error: If SQL checks fail.
    """
    required_files = [
        config.raw_dir / "merchants.csv",
        config.raw_dir / "sellers.csv",
        config.raw_dir / "customers.csv",
        config.raw_dir / "transactions.csv",
        config.duckdb_path,
    ]
    missing = [str(path) for path in required_files if not path.exists()]

    report: dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "status": "fail",
        "missing_files": missing,
        "checks": {},
        "dataset_hash": None,
    }

    if missing:
        report["checks"] = {"required_files": "missing"}
        return False, report

    suffix_ok, suffix_msg = _check_timestamp_suffix(config.raw_dir, config.expected_timezone_suffix)
    sql_checks = _run_duckdb_checks(config)

    tx_counts = sql_checks["tx_count_by_merchant"]
    tx_count_ok = all(
        abs(item["tx_count"] - config.expected_tx_per_merchant) <= 20 for item in tx_counts
    )

    checks = {
        "merchant_count": sql_checks["merchant_count"] == config.expected_merchants,
        "sellers_missing_refs": sql_checks["sellers_missing_refs"] == 0,
        "customers_missing_refs": sql_checks["customers_missing_refs"] == 0,
        "invalid_amount_rows": sql_checks["invalid_amount_rows"] == 0,
        "invalid_currency_rows": sql_checks["invalid_currency_rows"] == 0,
        "duplicate_transaction_ids": sql_checks["duplicate_transaction_ids"] == 0,
        "transactions_without_seller": sql_checks["transactions_without_seller"] == 0,
        "transaction_count_by_merchant": tx_count_ok,
        "timestamp_timezone_suffix": suffix_ok,
    }

    dataset_hash = _dataset_hash(config.raw_dir)
    report["dataset_hash"] = dataset_hash
    report["checks"] = {
        "pass_flags": checks,
        "details": {
            "timezone_check": suffix_msg,
            "sql_summary": sql_checks,
            "expected_currency": config.expected_currency,
            "expected_merchants": config.expected_merchants,
            "expected_tx_per_merchant": config.expected_tx_per_merchant,
        },
    }

    all_passed = all(checks.values())
    report["status"] = "pass" if all_passed else "fail"
    return all_passed, report


def main() -> None:
    """CLI entrypoint for dataset validation.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If argparse parsing fails.
    """
    parser = argparse.ArgumentParser(description="Validate synthetic DeUna MVP dataset")
    parser.add_argument(
        "--report-file",
        type=str,
        default="data/raw/validation_report.json",
        help="Path to write the validation report JSON.",
    )
    args = parser.parse_args()

    config = _load_config()
    ok, report = validate_dataset(config)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if ok:
        print("Validation passed.")
        print(f"Dataset hash: {report['dataset_hash']}")
    else:
        print("Validation failed.")
        print(f"See report: {report_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
