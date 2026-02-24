from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()

# Enable CORS (for GitHub Pages frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample NIFTY list (expand later)
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS"
]

# ----------------------------
# Feature Engineering
# ----------------------------
def compute_features(df):

    # Fix MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20d"] = df["log_return_1d"].rolling(20).std()
    df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1

    df = df.dropna()

    return df


# ----------------------------
# Ranking Logic
# ----------------------------
def predict_scores():

    results = []

    for ticker in NIFTY_50:
        try:
            df = yf.download(
                ticker,
                start="2023-01-01",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 60:
                print(f"Skipping {ticker} - insufficient data")
                continue

            df = compute_features(df)

            if df.empty:
                print(f"Skipping {ticker} - no features")
                continue

            latest = df.iloc[-1].copy()

            score = (
                float(latest["momentum_20d"]) * 0.6
                - float(latest["volatility_20d"]) * 0.4
            )

            results.append({
                "ticker": ticker,
                "score": round(score, 6)
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["ticker", "score"])

    ranking_df = pd.DataFrame(results)

    ranking_df = ranking_df.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df["rank"] = ranking_df.index + 1

    ranking_df["percentile"] = (
        100 * (1 - ranking_df.index / len(ranking_df))
    ).round(2)

    return ranking_df


# ----------------------------
# API Endpoint
# ----------------------------
@app.get("/rank")
def rank_stocks():

    ranking_df = predict_scores()

    if ranking_df.empty:
        return {
            "status": "error",
            "message": "No data available",
            "ranking": []
        }

    return {
        "status": "success",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_stocks": len(ranking_df),
        "ranking": ranking_df.to_dict(orient="records")
    }