-- ============================================================
-- Zepto Data & AI Platform
-- Data Pipeline Module
-- Required SQL Analysis Queries
-- ============================================================

-- Query 1: Book count by category
-- Demonstrates GROUP BY and JOIN.
SELECT
    c.category_name,
    COUNT(b.book_id) AS book_count
FROM categories c
JOIN books b
    ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY book_count DESC;


-- Query 2: Average price by category
-- Demonstrates AVG aggregation and GBP/INR analysis.
SELECT
    c.category_name,
    ROUND(AVG(b.price_inr), 2) AS average_price_inr
FROM categories c
JOIN books b
    ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY average_price_inr DESC;


-- Query 3: Rating distribution
-- Demonstrates GROUP BY over the rating field.
SELECT
    rating,
    COUNT(*) AS book_count
FROM books
GROUP BY rating
ORDER BY rating;


-- Query 4: Top 5 most expensive books
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
-- Demonstrates COUNT, MIN, MAX, AVG and GROUP BY.
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
