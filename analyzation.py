from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Movie_Streaming_Clean_500.csv"


df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates()
df.columns = df.columns.str.strip()

text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].str.strip()

df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

numeric_columns = [
    "Released_Year",
    "Runtime",
    "IMDB_Rating",
    "Meta_score",
    "No_of_Votes",
    "Gross",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

df = df[df["IMDB_Rating"].between(0, 10)]
df = df[df["Meta_score"].between(0, 100)]

print("=" * 55)
print("MOVIE STREAMING DATA ANALYSIS")
print("=" * 55)

print("\n1. Dataset Overview")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

print("\n2. Dataset Summary")
print(df[numeric_columns].describe().round(2).to_string())

print("\n3. Top 10 Movies by IMDb Rating")
top_movies = df.nlargest(10, "IMDB_Rating")[
    ["Series_Title", "Released_Year", "Primary_Genre",
     "IMDB_Rating", "No_of_Votes"]
]
print(top_movies.to_string(index=False))

print("\n4. Top 10 Primary Genres")
print(df["Primary_Genre"].value_counts().head(10).to_string())

print("\n5. Average IMDb Rating by Primary Genre")
genre_ratings = (
    df.groupby("Primary_Genre")["IMDB_Rating"]
    .mean()
    .sort_values(ascending=False)
)
print(genre_ratings.head(10).round(2).to_string())

print("\n6. Highest-Grossing Movies")
top_grossing = df.nlargest(10, "Gross")[
    ["Series_Title", "Released_Year", "Gross"]
]
top_grossing["Gross"] = top_grossing["Gross"].apply(
    lambda value: f"${value:,.0f}"
)
print(top_grossing.to_string(index=False))

print("\n7. Average Runtime by Primary Genre")
genre_runtime = (
    df.groupby("Primary_Genre")["Runtime"]
    .mean()
    .sort_values(ascending=False)
)
print(genre_runtime.head(10).round(2).to_string())

print("\n8. Average Gross by Primary Genre")
genre_gross = (
    df.groupby("Primary_Genre")["Gross"]
    .mean()
    .sort_values(ascending=False)
)
genre_gross = genre_gross.apply(lambda value: f"${value:,.0f}")
print(genre_gross.head(10).to_string())

print("\n9. Movies Released by Year")
year_counts = df.groupby("Released_Year").size().sort_index()
print(year_counts.tail(10).to_string())

print("\n10. Numeric Correlations")
correlations = df[numeric_columns].corr()
print(correlations.round(3).to_string())