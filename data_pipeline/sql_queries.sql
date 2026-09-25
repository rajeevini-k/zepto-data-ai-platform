-- ============================================================
-- Zepto Data & AI Platform
-- Data Pipeline Module
-- Required SQL Analysis Queries
-- ============================================================


-- Query 1: Distinct categories
-- Demonstrates SELECT, DISTINCT, and ORDER BY.
SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;


-- Query 2: Books in a price range
-- Demonstrates WHERE and BETWEEN.
SELECT
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock
FROM books
WHERE price_inr BETWEEN 2000 AND 5000
ORDER BY price_inr DESC;


-- Query 3: Books with selected ratings
-- Demonstrates WHERE and IN.
SELECT
    title,
    rating,
    price_inr
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_inr DESC
LIMIT 10;


-- Query 4: Top 5 most expensive books by category
-- Demonstrates JOIN, ORDER BY, and LIMIT.
SELECT
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.price_inr DESC
LIMIT 5;


-- Query 5: Category-level price aggregation
-- Demonstrates JOIN, COUNT, MIN, MAX, AVG, GROUP BY, and ORDER BY.
SELECT
    c.category_name,
    COUNT(b.book_id) AS book_count,
    ROUND(MIN(b.price_inr), 2) AS minimum_price_inr,
    ROUND(MAX(b.price_inr), 2) AS maximum_price_inr,
    ROUND(AVG(b.price_inr), 2) AS average_price_inr
FROM categories c
JOIN books b
    ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY average_price_inr DESC;
