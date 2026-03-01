from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI()

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clickneeth.github.io"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# -----------------------------
# NIFTY 50 LIST
# -----------------------------
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

# -----------------------------
# GLOBAL CACHE
# -----------------------------
cached_ranking = None
last_computed_date = None


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def compute_features(df):
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20d"] = df["log_return"].rolling(20).std()
    df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1
    df = df.dropna()
    return df


# -----------------------------
# SAFE VALUE EXTRACTOR
# -----------------------------
def safe_float(value):
    """
    Converts pandas scalar/Series to float safely.
    """
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


# -----------------------------
# RANKING ENGINE (FULL SAFE)
# -----------------------------
def generate_ranking():

    results = []

    for ticker in NIFTY_50:
        try:
            df = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 40:
                continue

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = compute_features(df)

            if df.empty:
                continue

            latest = df.iloc[-1]

            momentum = safe_float(latest["momentum_20d"])
            volatility = safe_float(latest["volatility_20d"])

            if momentum is None or volatility is None:
                continue

            score = (momentum * 0.6) - (volatility * 0.4)

            results.append({
                "ticker": ticker,
                "score": round(score, 6)
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if len(results) == 0:
        print("⚠️ No valid stock data fetched.")
        return None

    ranking_df = pd.DataFrame(results)

    ranking_df = ranking_df.sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df["rank"] = ranking_df.index + 1

    ranking_df["percentile"] = (
        100 * (1 - ranking_df.index / len(ranking_df))
    ).round(2)

    return ranking_df


# -----------------------------
# DAILY CACHE (RETRY + STICKY)
# -----------------------------
def get_cached_ranking():
    global cached_ranking
    global last_computed_date

    today = datetime.now().date()

    if cached_ranking is None or last_computed_date != today:

        print("🔄 Recomputing ranking...")

        ranking_df = None

        # Retry 3 times
        for attempt in range(3):
            print(f"Attempt {attempt + 1}")
            ranking_df = generate_ranking()
            if ranking_df is not None:
                break

        if ranking_df is not None:

            cached_ranking = {
                "status": "success",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "total_stocks": len(ranking_df),
                "ranking": ranking_df.to_dict(orient="records")
            }

            last_computed_date = today
            print("✅ Ranking computed successfully.")

        else:
            print("⚠️ Ranking failed after retries.")

            if cached_ranking is not None:
                print("Using previous cached ranking.")
                return cached_ranking

            return {
                "status": "error",
                "message": "Data temporarily unavailable. Please try again shortly.",
                "ranking": [],
                "total_stocks": 0,
                "last_updated": None
            }

    return cached_ranking


# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.get("/rank")
def rank_stocks():
    return get_cached_ranking()


@app.get("/")
def home():
    return {"message": "NIFTY 50 Alpha Engine is live"}