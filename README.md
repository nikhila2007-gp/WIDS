# 📈 ChronosTrade

**Multi-Modal Stock Market Prediction using ARIMA and LSTM**

## 🔍 Project Overview

ChronosTrade is a time-series forecasting project that predicts stock prices using two different approaches:

* **ARIMA** – a classical statistical time-series model
* **LSTM** – a deep learning model capable of capturing non-linear patterns

The project compares both models to understand their strengths, limitations, and performance on real stock market data.

---

## 📊 Dataset

* **Source:** Yahoo Finance
* **Data Used:** Historical stock price data
* **Features:** Date, Open, High, Low, Close, Adjusted Close, Volume
* **Target Variable:** Adjusted Closing Price
* **Time Range:**

  * ARIMA: 1–2 years
  * LSTM: 3–5 years

---

## 🧠 Models Implemented

### 1️⃣ ARIMA (AutoRegressive Integrated Moving Average)

* Stationarity tested using **ADF Test**
* Differencing applied where required
* Parameters (p, d, q) selected using **ACF & PACF plots**
* Evaluated using MAE, MSE, RMSE

### 2️⃣ LSTM (Long Short-Term Memory)

* Data normalized using **MinMaxScaler**
* Sliding window technique for supervised learning
* Built using **TensorFlow/Keras**
* Evaluated using MAE, RMSE, MAPE

---

## 📈 Evaluation & Comparison

Both models are evaluated on the same test dataset and compared based on:

* Prediction accuracy
* Ability to capture trends and non-linear patterns
* Computational complexity
* Model assumptions

---

## 📁 Repository Structure

```
ChronosTrade/
│
├── data/                 # Stock price datasets
├── notebooks/            # Jupyter notebooks for ARIMA & LSTM
│   ├── week1_arima.ipynb
│   └── week2_lstm.ipynb
├── reports/              # Written reports and analysis
├── README.md             # Project overview
└── requirements.txt      # Python dependencies
```

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy, Matplotlib
* Statsmodels (ARIMA)
* TensorFlow / Keras (LSTM)
* Scikit-learn

---

## 📌 Key Learnings

* Importance of stationarity in time-series forecasting
* Differences between statistical and deep learning models
* ARIMA performs well on linear patterns, while LSTM captures complex non-linear trends
* Model selection depends on data size, complexity, and interpretability needs

---

## 🚀 Future Enhancements

* Add **sentiment analysis** using news and social media data
* Use **technical indicators** as additional features
* Explore **Bidirectional LSTM** and attention mechanisms


