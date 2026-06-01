# Political Sentiment Analysis

A machine learning project to classify political sentiment from tweets related to the 2024 US Presidential Election candidates — across party lines and engagement metrics.

---

## Project Overview

This project analyzes tweet-level sentiment (positive, neutral, negative) towards five 2024 US presidential candidates using classical ML classification models. The pipeline covers end-to-end data preprocessing, feature engineering, and model comparison.

**Candidates covered:**
- Kamala Harris (Democratic Party)
- Donald Trump (Republican Party)
- Jill Stein (Green Party)
- Robert Kennedy (Independent)
- Chase Oliver (Libertarian Party)

---

## Repository Structure

political-sentiment-analysis  
 ┣ Political_Sentiment_Analysis.ipynb — Full ML pipeline notebook  
 ┗ Political_Sentiment_Analysis.csv — Dataset (500 tweets)

---

## Dataset

| Feature | Description |
|---|---|
| `tweet_id` | Unique tweet identifier |
| `user_handle` | Twitter/X username |
| `timestamp` | Date and time of tweet |
| `tweet_text` | Raw tweet content |
| `candidate` | Candidate the tweet mentions |
| `party` | Candidate's political party |
| `retweets` | Retweet count |
| `likes` | Like count |
| `sentiment` | Target label: `positive`, `neutral`, `negative` |

- **Total records:** 500 tweets
- **Class distribution:** ~66% positive, ~25% neutral, ~9% negative

---

## ML Pipeline

### 1. Data Preprocessing
- Null value imputation (mean for numerical, mode for categorical)
- Duplicate removal
- One-Hot Encoding for `candidate`, `party`, `sentiment`
- IQR-based outlier treatment
- StandardScaler feature normalization

### 2. Feature Selection
- `SelectKBest` with ANOVA F-score (`f_classif`)
- Top 10 features selected: retweets, likes, and encoded candidate/party columns

### 3. Train-Test Split
- 70% training / 30% testing (`random_state=4`)

### 4. Models Trained

| Model | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model` |
| K-Nearest Neighbors | `sklearn.neighbors` |
| Decision Tree | `sklearn.tree` |
| Random Forest | `sklearn.ensemble` |
| Support Vector Machine | `sklearn.svm` |
| Gradient Boosting | `sklearn.ensemble` |
| Naive Bayes | `sklearn.naive_bayes` |

---

## Author

**Jaishri Vijayakumar**  
B.Sc. Data Science | PSGR Krishnammal College for Women, Coimbatore  
LinkedIn: linkedin.com/in/jaishri-vijayakumar  
GitHub: github.com/jaishrivijayakumar
