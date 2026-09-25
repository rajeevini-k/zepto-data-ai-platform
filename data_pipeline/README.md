# Data Pipeline Module

## Overview

This module implements a complete raw-to-relational data pipeline for book catalog data from Books to Scrape.

The pipeline:

1. Scrapes book data using `requests` and `BeautifulSoup`.
2. Extracts data from six book categories.
3. Cleans and converts the scraped fields into usable Python types.
4. Converts GBP prices to INR using the required fixed project rate.
5. Stores the cleaned data in a normalized SQLite database.
6. Executes SQL analysis queries against the database.
7. Reads SQL results into pandas using `pd.read_sql()`.
8. Reproduces the SQL JOIN using `pd.merge()`.

## Dataset

Source: Books to Scrape

https://books.toscrape.com/

The pipeline currently scrapes these six categories:

- Travel
- Mystery
- Historical Fiction
- Sequential Art
- Classics
- Philosophy

The resulting dataset contains:

- 101 books
- 6 categories

This exceeds the project requirement of at least 60 books across at least 3 categories.

## Extracted Fields

The raw scraper captures:

- `title`
- `price_gbp`
- `star_rating`
- `availability`
- `category`

## Cleaning Decisions

### Price

The currency symbol is removed and the price is converted to a floating-point value.

Example: `£45.17` becomes `45.17`.

### Rating

The text ratings from the website are converted into integers:

- One -> 1
- Two -> 2
- Three -> 3
- Four -> 4
- Five -> 5

### Availability

The availability text is converted into a Boolean `in_stock` field.

`In stock` becomes `True`.

### Missing and Duplicate Data

The scraped dataset was checked for missing values and duplicate rows.

The final dataset contains:

- 101 rows
- No missing values
- No duplicate rows

## Currency Conversion

The project requires the fixed conversion:

**1 GBP = 105.50 INR**

This is a fixed project-defined baseline and is not a live exchange rate.

The `price_inr` column is calculated as:

`price_inr = price_gbp * 105.50`

No external currency API is required.

## Output Dataset

The cleaned dataset is stored at:

`data_pipeline/books_clean.csv`

Important columns include:

- `title`
- `price_gbp`
- `rating`
- `in_stock`
- `category`
- `price_inr`

## SQLite Database

The SQLite database is stored at:

`data_pipeline/books.db`

The database uses a normalized two-table design.

### categories

- `category_id INTEGER PRIMARY KEY AUTOINCREMENT`
- `category_name TEXT NOT NULL UNIQUE`

### books

- `book_id INTEGER PRIMARY KEY AUTOINCREMENT`
- `title TEXT NOT NULL`
- `price_gbp REAL NOT NULL`
- `price_inr REAL NOT NULL`
- `rating INTEGER NOT NULL`
- `in_stock INTEGER NOT NULL`
- `category_id INTEGER NOT NULL`
- Foreign key: `category_id REFERENCES categories(category_id)`

Foreign-key validation was performed and resulted in zero violations.

## SQL Analysis

The SQL queries are stored in:

`data_pipeline/sql_queries.sql`

The executed query outputs are stored in:

`data_pipeline/sql_outputs.txt`

Five SQL queries are included. Collectively they demonstrate:

- SELECT
- LIMIT
- DISTINCT
- WHERE
- ORDER BY
- BETWEEN
- JOIN
- COUNT
- MIN
- MAX
- AVG

## Pandas / SQL Validation

SQL query results were read back into pandas using `pd.read_sql()`.

The database JOIN was independently reproduced using `pd.merge()`.

The SQL JOIN result and pandas merge result were compared and found to have equivalent columns and values.

## Reproducible Pipeline

The complete pipeline is implemented in:

`data_pipeline/data_pipeline.py`

Run from the repository root:

```bash
python data_pipeline/data_pipeline.py
```

The pipeline validates:

- Minimum number of books
- Minimum number of categories
- Orphaned foreign-key records
- Foreign-key violations

## Files

```text
data_pipeline/
├── README.md
├── books.db
├── books_clean.csv
├── data_pipeline.py
├── sql_queries.sql
└── sql_outputs.txt
```

## Design Decisions

The pipeline uses a normalized relational structure so that category information is stored once in the `categories` table and referenced by books through `category_id`.

The fixed GBP-to-INR conversion required by the project is applied during the cleaning and enrichment stage before database loading.

The pipeline is implemented as a reproducible Python script rather than relying on manual data preparation.