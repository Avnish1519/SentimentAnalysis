import pandas as pd
import os

# Load the CSV file without headers
df = pd.read_csv("sentiment.csv", header=None)

# Assign the correct column names based on the dataset structure
df.columns = [
    "tweet_id", "airline_sentiment", "airline_sentiment_confidence",
    "negativereason", "negativereason_confidence", "airline",
    "name", "tweet_coord", "tweet_location", "user_timezone",
    "text", "tweet_created", "retweet_count", "tweet_id2", "user_handle"
]

# Keep only the relevant columns
df = df[["text", "airline_sentiment"]]

# Drop any rows with missing values
df.dropna(inplace=True)

# Create an output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save the full cleaned data
df.to_csv("cleaned_sentiment.csv", index=False)

# Split and save the data based on sentiment
for sentiment in df["airline_sentiment"].unique():
    filtered = df[df["airline_sentiment"] == sentiment]
    out_path = f"outputs/{sentiment.lower()}_reviews.csv"
    filtered.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(filtered)} rows.")

import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="airline_sentiment", data=df, order=["negative", "neutral", "positive"])
plt.title("Distribution of Sentiment Labels")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("outputs/sentiment_distribution.png")
plt.close()
print("Saved outputs/sentiment_distribution.png (class distribution plot)")


import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Clean the text more aggressively
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"[^a-z\s]", "", text) # Keep letters and spaces only
    text = re.sub(r"\s+", " ", text)     # Remove extra spaces
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["airline_sentiment"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()



