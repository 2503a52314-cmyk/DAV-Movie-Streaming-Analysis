from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Movie_Streaming_Clean_500.csv"
CLEANED_FILE = BASE_DIR / "Movie_Streaming_Cleaned.csv"
OUTPUT_FILE = BASE_DIR / "Movie_Streaming_10_Visualizations.png"

sns.set_theme(style="whitegrid")

# Load and clean the dataset
df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates()
df.columns = df.columns.str.strip()

text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].str.strip()

df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

numeric_columns = ["Released_Year", "Runtime", "IMDB_Rating",
                   "Meta_score", "No_of_Votes", "Gross"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(include=["object", "string"]).columns
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

df = df[df["IMDB_Rating"].between(0, 10)]
df = df[df["Meta_score"].between(0, 100)]
df.to_csv(CLEANED_FILE, index=False)

# Create a 5 x 2 dashboard containing exactly 10 visualizations
fig, axes = plt.subplots(5, 2, figsize=(18, 25))
fig.suptitle("Movie Streaming Analysis - 10 Visualizations",
             fontsize=22, fontweight="bold", y=1.01)
axes = axes.flatten()

# 1. IMDb rating distribution
sns.histplot(df["IMDB_Rating"], bins=15, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("1. Distribution of IMDb Ratings")
axes[0].set_xlabel("IMDb Rating")

# 2. Runtime distribution
sns.histplot(df["Runtime"], bins=20, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("2. Distribution of Runtime")
axes[1].set_xlabel("Runtime (minutes)")

# 3. Top primary genres
top_genres = df["Primary_Genre"].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, ax=axes[2],
            hue=top_genres.index, palette="viridis", legend=False)
axes[2].set_title("3. Top 10 Primary Genres")
axes[2].set_xlabel("Number of Movies")

# 4. Certificates
certificate_counts = df["Certificate"].value_counts().head(10)
sns.barplot(x=certificate_counts.index, y=certificate_counts.values,
            ax=axes[3], hue=certificate_counts.index, palette="mako", legend=False)
axes[3].set_title("4. Movie Count by Certificate")
axes[3].set_xlabel("Certificate")
axes[3].tick_params(axis="x", rotation=30)

# 5. Average IMDb rating by primary genre
genre_ratings = (df.groupby("Primary_Genre")["IMDB_Rating"]
                 .mean()
                 .sort_values(ascending=False)
                 .head(10))
sns.barplot(x=genre_ratings.values, y=genre_ratings.index, ax=axes[4],
            hue=genre_ratings.index, palette="crest", legend=False)
axes[4].set_title("5. Highest Average IMDb Rating by Genre")
axes[4].set_xlabel("Average IMDb Rating")
axes[4].set_xlim(0, 10)

# 6. Median gross by primary genre
genre_gross = (df.groupby("Primary_Genre")["Gross"]
               .median()
               .sort_values(ascending=False)
               .head(10) / 1_000_000)
sns.barplot(x=genre_gross.values, y=genre_gross.index, ax=axes[5],
            hue=genre_gross.index, palette="rocket", legend=False)
axes[5].set_title("6. Median Gross by Genre")
axes[5].set_xlabel("Median Gross (USD millions)")

# 7. IMDb rating vs number of votes
sns.scatterplot(data=df, x="No_of_Votes", y="IMDB_Rating",
                hue="Primary_Genre", palette="tab10", alpha=0.75,
                ax=axes[6], legend=False)
axes[6].set_title("7. IMDb Rating vs Number of Votes")
axes[6].set_xlabel("Number of Votes")
axes[6].ticklabel_format(axis="x", style="plain")

# 8. Runtime vs IMDb rating
sns.scatterplot(data=df, x="Runtime", y="IMDB_Rating",
                hue="Primary_Genre", palette="tab10", alpha=0.75,
                ax=axes[7], legend=False)
axes[7].set_title("8. Runtime vs IMDb Rating")
axes[7].set_xlabel("Runtime (minutes)")

# 9. Number of movies released per year
year_counts = df.groupby("Released_Year").size()
axes[8].plot(year_counts.index, year_counts.values,
             marker="o", linewidth=1.5, markersize=3, color="#C44E52")
axes[8].set_title("9. Movies Released per Year")
axes[8].set_xlabel("Released Year")
axes[8].set_ylabel("Number of Movies")

# 10. Numeric correlation heatmap
correlation = df[["Released_Year", "Runtime", "IMDB_Rating",
                  "Meta_score", "No_of_Votes", "Gross"]].corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=axes[9])
axes[9].set_title("10. Numeric Feature Correlations")

plt.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Data cleaning completed.")
print(f"Cleaned dataset: {CLEANED_FILE}")
print(f"10-graph visualization: {OUTPUT_FILE}")
print(f"Dataset shape: {df.shape}")
