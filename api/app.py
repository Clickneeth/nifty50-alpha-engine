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

# ----------------------------
# FULL NIFTY 50 UNIVERSE
# ----------------------------
NIFTY_50 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS",
    "GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS",
    "INDUSINDBK.NS","INFY.NS","ITC.NS","JSWSTEEL.NS",
    "KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS",
    "RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHREECEM.NS",
    "SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS",
    "UPL.NS","WIPRO.NS"
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
                period="6mo",       # faster than full history
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 40:
                print(f"Skipping {ticker} - insufficient data")
                continue

            df = compute_features(df)

            if df.empty:
                print(f"Skipping {ticker} - no features")
                continue

            latest = df.iloc[-1]

            momentum = float(latest["momentum_20d"])
            volatility = float(latest["volatility_20d"])

            score = (momentum * 0.6) - (volatility * 0.4)

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