"""
Zepto Data & AI Platform
Data Pipeline Module

Scrapes books from Books to Scrape, cleans and enriches the data,
converts GBP prices to INR using the required fixed exchange rate,
and loads the result into a normalized SQLite database.
"""

from pathlib import Path
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_PIPELINE_DIR = REPO_DIR / "data_pipeline"

CSV_PATH = DATA_PIPELINE_DIR / "books_clean.csv"
DB_PATH = DATA_PIPELINE_DIR / "books.db"


# Required categories
CATEGORIES = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art": "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
    "Classics": "https://books.toscrape.com/catalogue/category/books/classics_6/index.html",
    "Philosophy": "https://books.toscrape.com/catalogue/category/books/philosophy_7/index.html",
}


# ============================================================
# SCRAPING
# ============================================================

def scrape_category(category_name, category_url):
    """
    Scrape all books from one Books to Scrape category.
    """

    response = requests.get(
        category_url,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.content,
        "html.parser"
    )

    books = []

    for article in soup.select("article.product_pod"):

        title_tag = article.select_one("h3 a")
        price_tag = article.select_one(".price_color")
        availability_tag = article.select_one(".availability")
        rating_tag = article.select_one("p.star-rating")

        if not all([
            title_tag,
            price_tag,
            availability_tag,
            rating_tag
        ]):
            continue

        title = title_tag.get("title", "").strip()

        price = price_tag.get_text(strip=True)

        availability = availability_tag.get_text(
            " ",
            strip=True
        )

        rating_classes = rating_tag.get("class", [])

        rating_words = [
            value
            for value in rating_classes
            if value != "star-rating"
        ]

        star_rating = (
            rating_words[0]
            if rating_words
            else None
        )

        books.append(
            {
                "title": title,
                "price_gbp": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name,
            }
        )

    return books


def scrape_all_categories():
    """
    Scrape all configured categories.
    """

    all_books = []

    for category_name, category_url in CATEGORIES.items():

        category_books = scrape_category(
            category_name,
            category_url
        )

        print(
            f"{category_name}: "
            f"{len(category_books)} books"
        )

        all_books.extend(category_books)

    return pd.DataFrame(all_books)


# ============================================================
# CLEANING
# ============================================================

def parse_price(value):
    """
    Convert a value such as '£45.17' to float.
    """

    try:
        if pd.isna(value):
            return None

        value = str(value).strip()

        value = (
            value
            .replace("£", "")
            .replace("Â£", "")
            .strip()
        )

        return float(value)

    except (ValueError, TypeError):
        return None


def clean_data(raw_df):
    """
    Clean and enrich the scraped dataset.
    """

    df = raw_df.copy()

    # Clean GBP price
    df["price_gbp"] = (
        df["price_gbp"]
        .apply(parse_price)
    )

    # Convert ratings
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    df["rating"] = (
        df["star_rating"]
        .map(rating_map)
    )

    # Convert availability to boolean
    df["in_stock"] = (
        df["availability"]
        .str.strip()
        .str.lower()
        .eq("in stock")
    )

    # Remove rows where required fields
    # could not be parsed
    df = df.dropna(
        subset=[
            "title",
            "price_gbp",
            "rating",
            "category",
        ]
    ).copy()

    df["rating"] = (
        df["rating"]
        .astype(int)
    )

    # Fixed GBP -> INR conversion
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    return df


# ============================================================
# CSV OUTPUT
# ============================================================

def save_clean_csv(df):
    """
    Save cleaned dataset as CSV.
    """

    DATA_PIPELINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CSV_PATH,
        index=False
    )

    print(
        f"Saved cleaned dataset to: {CSV_PATH}"
    )


# ============================================================
# DATABASE
# ============================================================

def create_database():
    """
    Create normalized SQLite database.
    """

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "PRAGMA foreign_keys = ON;"
    )

    conn.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        );
        """
    )

    conn.commit()

    return conn


def load_database(df, conn):
    """
    Load categories and books into normalized tables.
    """

    # Insert categories
    categories = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )

    for category in categories["category"]:

        conn.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?);
            """,
            (category,)
        )

    conn.commit()

    # Build category lookup
    category_lookup = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories;
        """,
        conn
    )

    category_map = dict(
        zip(
            category_lookup["category_name"],
            category_lookup["category_id"]
        )
    )

    # Insert books
    book_rows = []

    for _, row in df.iterrows():

        book_rows.append(
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                int(category_map[row["category"]]),
            )
        )

    conn.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        book_rows
    )

    conn.commit()


# ============================================================
# VALIDATION
# ============================================================

def validate_database(conn):
    """
    Validate database row counts and referential integrity.
    """

    book_count = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM books;
        """,
        conn
    ).iloc[0]["count"]

    category_count = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM categories;
        """,
        conn
    ).iloc[0]["count"]

    orphaned_books = pd.read_sql(
        """
        SELECT COUNT(*) AS count
        FROM books b
        LEFT JOIN categories c
            ON b.category_id = c.category_id
        WHERE c.category_id IS NULL;
        """,
        conn
    ).iloc[0]["count"]

    foreign_key_violations = pd.read_sql(
        """
        PRAGMA foreign_key_check;
        """,
        conn
    )

    print("\n===================================")
    print("DATABASE VALIDATION")
    print("===================================")

    print("Books:", book_count)
    print("Categories:", category_count)
    print("Orphaned books:", orphaned_books)
    print(
        "Foreign-key violations:",
        len(foreign_key_violations)
    )

    assert book_count >= 60
    assert category_count >= 3
    assert orphaned_books == 0
    assert len(foreign_key_violations) == 0

    print("\nDatabase validation passed.")


# ============================================================
# MAIN
# ============================================================


def validate_sql_join_with_pandas_merge(conn, books_df):
    """
    Reproduce the normalized SQL JOIN using pandas.merge()
    and verify that both approaches return equivalent data.
    """

    sql_join = pd.read_sql(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            b.category_id,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
        """,
        conn
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        ORDER BY category_id
        """,
        conn
    )

    books_for_merge = books_df.copy()

    # Add the database book_id values in the same deterministic order.
    books_for_merge.insert(
        0,
        "book_id",
        range(1, len(books_for_merge) + 1)
    )

    pandas_merge = pd.merge(
        books_for_merge[
            [
                "book_id",
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock"
            ]
        ],
        books_for_merge.assign(
            category_id=books_for_merge["category"].map(
                dict(
                    zip(
                        categories_df["category_name"],
                        categories_df["category_id"]
                    )
                )
            )
        )[
            ["book_id", "category_id", "category"]
        ],
        on="book_id",
        how="inner"
    )

    pandas_merge = pandas_merge[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_id",
            "category"
        ]
    ].rename(columns={"category": "category_name"})

    pandas_merge = pandas_merge.sort_values(
        "book_id"
    ).reset_index(drop=True)

    sql_join = sql_join.sort_values(
        "book_id"
    ).reset_index(drop=True)

    same_shape = sql_join.shape == pandas_merge.shape

    same_columns = list(sql_join.columns) == list(
        pandas_merge.columns
    )

    if same_shape and same_columns:
        sql_compare = sql_join.copy()
        pandas_compare = pandas_merge.copy()

        sql_compare["in_stock"] = sql_compare["in_stock"].astype(int)
        pandas_compare["in_stock"] = pandas_compare["in_stock"].astype(int)

        same_values = sql_compare.equals(pandas_compare)
    else:
        same_values = False

    print("\n===================================")
    print("SQL JOIN VS PANDAS MERGE")
    print("===================================")
    print("SQL JOIN shape:", sql_join.shape)
    print("Pandas merge shape:", pandas_merge.shape)
    print("Same shape:", same_shape)
    print("Same columns:", same_columns)
    print("Same values:", same_values)
    print("Overall SQL JOIN == pandas merge:", same_shape and same_columns and same_values)

    return same_shape and same_columns and same_values



def main():

    print("===================================")
    print("ZEPTO DATA PIPELINE")
    print("===================================")

    # Scrape
    raw_df = scrape_all_categories()

    print(
        f"\nTotal scraped books: {len(raw_df)}"
    )

    # Clean
    clean_df = clean_data(raw_df)

    print(
        f"Total cleaned books: {len(clean_df)}"
    )

    # Save CSV
    save_clean_csv(clean_df)

    # Create database
    conn = create_database()

    try:

        # Load database
        load_database(
            clean_df,
            conn
        )

        # Validate
        validate_database(conn)

        # Validate SQL JOIN against pandas merge
        validate_sql_join_with_pandas_merge(
            conn,
            clean_df
        )

    finally:

        conn.close()

    print("\n===================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("===================================")


if __name__ == "__main__":
    main()
