# Data Package Overview

## Purpose
This package is responsible for building the synthetic transactional dataset used by the backend MVP.
Its goal is to produce realistic but controlled merchant data that is easy to inspect, easy to regenerate, and easy to validate.

## What the generator produces
The generator creates four core tables:
- merchants
- sellers
- customers
- transactions

The outputs are saved in two formats:
- CSV files for easy manual inspection
- A DuckDB database for fast analytical queries

It also writes a metadata file that records the exact generation settings and injected anomalies.

## Main design principles
The dataset is designed to be:
- Reproducible: the same seed creates the same data
- Simple: minimal entities and fields for the MVP
- Realistic enough: includes customer behavior patterns, seller attribution, seasonality, and controlled anomalies
- Testable: includes deterministic patterns that can be detected in evaluations

## How data generation works
The generation flow is deterministic and follows these steps:

1. Read configuration from environment values
The generator reads parameters such as seed, number of merchants, history window, transaction target per merchant, end date, timezone, currency, and output paths.

2. Build merchants
Merchants are created with fixed IDs and simple business metadata such as city and category.

3. Build sellers
Each merchant gets a seller structure with role variety.
Some merchants have only an owner, while others include one or two additional sellers.
This enables seller-performance analysis.

4. Build customers
Customers are generated per merchant with:
- stable anonymized customer IDs
- customer display names for more human-friendly outputs
- masked document and account references
- a frequency profile that drives visit behavior
- first seen date

5. Build transaction demand by day
Daily demand is weighted to simulate practical patterns:
- weekday and weekend differences
- mild month-level variation
- deterministic anomaly windows for selected merchants

6. Build transactions
For each transaction, the generator samples:
- merchant
- seller (always present)
- customer
- day and hour
- amount

Each transaction includes timestamp with timezone offset, amount, currency, payer masked references, payment channel, and sequence fields.

7. Write outputs
The generated tables are written to CSV and loaded into DuckDB tables.
The metadata file records seed, time assumptions, and anomaly configuration.

## Reproducibility model
Reproducibility is based on a single random seed and fixed generation rules.
If all configuration values are unchanged, regeneration should produce the same logical dataset and deterministic validation result.

## How validation is handled
Validation is implemented in the companion validation module at [src/agent/validation/validate_dataset.py](src/agent/validation/validate_dataset.py).

Validation checks include:
- Required files exist
- Timestamp offset format is correct for the expected timezone
- Merchant count matches expectation
- Every transaction references a valid seller and customer
- Amount values are positive
- Currency matches expectation
- Transaction IDs are unique
- Seller attribution is present in all transactions
- Transaction count per merchant is near the configured target

Validation also computes and stores a dataset hash for traceability.

## Validation output
Validation writes a report with:
- pass or fail status
- detailed check results
- summary metrics
- dataset hash

This report is intended to make failures easy to diagnose without opening source code.

## Assumptions and intentional limits
This package currently assumes:
- One payment channel for all transactions
- No refunds or reversals
- Three merchants by default, with configurable count
- A fixed timezone and currency provided through configuration

The package is intentionally MVP-scoped and focused on income, customers, time patterns, and seller attribution.

## How to extend safely
If new fields or entities are added, keep these rules:
- Preserve deterministic generation with the same seed behavior
- Keep outputs inspectable in CSV
- Keep validation strict and updated with any schema changes
- Prefer simple, explicit assumptions over implicit behavior
- Document every new business rule in this README

## Related files
- [src/agent/data/generate_dataset.py](src/agent/data/generate_dataset.py)
- [src/agent/validation/validate_dataset.py](src/agent/validation/validate_dataset.py)
