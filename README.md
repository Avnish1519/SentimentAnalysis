# ✈️ Twitter Sentiment Analysis Project

This project performs **sentiment analysis** on Twitter data related to airlines. It classifies tweets into **positive**, **neutral**, and **negative** sentiments using Natural Language Processing (NLP) techniques and machine learning models.

---

## 🔗 Repository Link

👉 [https://github.com/Avnish1519/SentimentAnalysis](https://github.com/Avnish1519/SentimentAnalysis)

✔️ This is a **public** repository and is accessible to all reviewers.

---

## 📁 Project Structure

SentimentAnalysisProject/
│
├── sentiment.csv # Raw dataset
├── cleaned_sentiment.csv # Cleaned dataset
├── sentiment_analysis.py # Main Python script
├── outputs/ # Output directory for results
│ ├── airline_sentiment_reviews.csv
│ ├── negative_reviews.csv
│ ├── neutral_reviews.csv
│ ├── positive_reviews.csv
│ ├── confusion_matrix.png
│ └── sentiment_distribution.png
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Requirements

Before running the project, install the required Python libraries:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
▶️ How to Run the Project
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Avnish1519/SentimentAnalysis.git
cd SentimentAnalysis
Run the analysis script:

bash
Copy
Edit
python sentiment_analysis.py
Outputs will be saved in the outputs/ folder.

📊 Outputs & Visuals
Sentiment Distribution Plot


Classification Report

markdown
Copy
Edit
Precision | Recall | F1-score | Support
---------------------------------------
 Negative |  0.80  |  0.94   |  0.86   | 1880
 Neutral  |  0.61  |  0.43   |  0.51   | 580
 Positive |  0.81  |  0.57   |  0.67   | 469
---------------------------------------
 Accuracy |               0.78         | 2929
Exported Datasets:

Cleaned sentiment dataset

Separate files for positive, neutral, and negative tweets

🙌 Credits
Dataset Source: Kaggle - Twitter US Airline Sentiment

Developed by: Avnish
