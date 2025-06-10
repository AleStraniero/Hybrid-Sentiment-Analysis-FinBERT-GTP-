
### Hybrid: FinBERT + GPT-4 Sentiment Analysis Pipeline
### Follows implementation by ProsusAI https://github.com/ProsusAI/finBERT
#  FinBERT gives primary sentiment score (pos‑prob − neg‑prob)
#  The FinBERT-onyl score is extracted for benchmarking
#  GPT‑4 called upon for fallback when FinBERT is effectively
#  Fallback activated when FinBERT score is neutral (|score| < GPT4_CONFIDENCE_THRESHOLD)
#  Carried out with Python 3.13

import os
import time 
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification

USE_GPT4_FALLBACK = True
GPT4_CONFIDENCE_THRESHOLD = 0.05                     
OPENAI_MODEL = "gpt-4o-mini" # GPT model choice supported by literature
OPENAI_MAX_TOKENS = 10

DERIVED_PATH = "derived"
Path(DERIVED_PATH).mkdir(exist_ok=True)

GPT4_CHECKPOINT_EVERY = 250   # save after every 250 GPT-4 calls

if USE_GPT4_FALLBACK: # Add API key to .env in same directory
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set.")
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

market_df = pd.read_csv("sp500_vix_data.csv", parse_dates=["Date"])
news_df = pd.read_csv("nyt_aggregated_data.csv", parse_dates=["Date"])
news_df.rename(columns={"Summary": "summary", "Headline": "headline", "Sector": "sector"}, inplace=True)

trading_days = set(market_df["Date"].dt.normalize()) # Aligning news dates with trading calendar
news_df = news_df[news_df["Date"].dt.normalize().isin(trading_days)].reset_index(drop=True)

FINBERT_MODEL = "ProsusAI/finbert" # FinBERT model - ProsusAI
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert.eval()

@torch.inference_mode()
def finbert_score(texts: List[str], batch_size: int = 32) -> Tuple[List[float], List[np.ndarray]]:
    scores, probs_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        out = finbert(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=1).cpu().numpy()
        batch_scores = probs[:,1] - probs[:,2]
        scores.extend(batch_scores.tolist())
        probs_all.extend(probs)
    return scores, probs_all

def gpt4_sentiment_single(text):
    """Call GPT-4 for sentiment fallback, return -1, 0, or 1."""        
    system_msg = "You are a financial news analyst."
    user_msg = (
        "Classify the sentiment of the given NY Times news article summary {summary}, which is closely related to the {sector} industry, as positive for buy, negative for sell, or neutral for hold position, for the US Stock market and provide the probability values for your classification."
        "Answer with just one word.\n\n"
        f"Summary: \"{text}\""
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=OPENAI_MAX_TOKENS,
            temperature=0.0,
        )
        ans = response.choices[0].message.content.strip().lower()
    except Exception as e:
        print("GPT-4 error (treated as neutral):", e)
        return 0.0
    if ans.startswith("pos"):
        return 1.0
    if ans.startswith("neg"):
        return -1.0
    return 0.0

def add_sentiment(news_df, use_gpt4=False):
    finbert_path = os.path.join(DERIVED_PATH, "nyt_with_finbert_sentiment.csv") # Checkpoint 1: If FinBERT file exists, skip FinBERT and load
    if os.path.exists(finbert_path):
        print("Checkpoint: FinBERT sentiment file found. Loading and skipping FinBERT step.")
        news_df = pd.read_csv(finbert_path, parse_dates=["Date"])
    else:
        print("Scoring FinBERT…")
        start_finbert = time.time()
        fin_scores, fin_probs = finbert_score(news_df["summary"].tolist())
        end_finbert = time.time()
        news_df["FinBERT_score"] = fin_scores
        news_df["FinBERT_prob_neu"] = [p[0] for p in fin_probs]
        news_df["FinBERT_prob_pos"] = [p[1] for p in fin_probs]
        news_df["FinBERT_prob_neg"] = [p[2] for p in fin_probs]
        news_df["FinBERT_confidence"] = news_df["FinBERT_score"].abs()
        news_df["Sentiment"] = news_df["FinBERT_score"]  # Start with FinBERT
        print(f"FinBERT sentiment scored in {end_finbert - start_finbert:.2f} seconds.")
        news_df.to_csv(finbert_path, index=False)
        print(f"Checkpoint: FinBERT sentiment saved to {finbert_path}")

    num_gpt4 = 0
    gpt4_path = os.path.join(DERIVED_PATH, "nyt_with_gpt4_fallback_sentiment.csv")

    if use_gpt4:
        if os.path.exists(gpt4_path): # If checkpoint exists, load and skip fallback
            print("Checkpoint: GPT-4 fallback file found. Loading and skipping fallback step.")
            news_df = pd.read_csv(gpt4_path, parse_dates=["Date"])
        else:
            print("Running GPT-4 fallback…")
            low_conf_mask = news_df["FinBERT_confidence"] < GPT4_CONFIDENCE_THRESHOLD
            num_gpt4 = low_conf_mask.sum()
            print(f"News needing GPT-4 fallback: {num_gpt4} of {len(news_df)}")
            start_gpt4 = time.time()
            checkpoint_counter = 0
            for idx_num, idx in enumerate(news_df[low_conf_mask].index):
                news_df.at[idx, "Sentiment"] = gpt4_sentiment_single(news_df.at[idx, "summary"])
                checkpoint_counter += 1
                if checkpoint_counter % GPT4_CHECKPOINT_EVERY == 0: # Intermediate checkpoint
                    news_df.to_csv(gpt4_path, index=False)
                    print(f"Checkpoint: Saved GPT-4 fallback at {checkpoint_counter} / {num_gpt4} GPT-4 calls.")
            end_gpt4 = time.time()
            print(f"GPT-4 fallback completed in {end_gpt4 - start_gpt4:.2f} seconds.")
            news_df.to_csv(gpt4_path, index=False)
            print(f"Checkpoint: Final GPT-4 fallback saved to {gpt4_path}")
    else:
        num_gpt4 = 0

    return news_df, num_gpt4
   
Path(DERIVED_PATH).mkdir(exist_ok=True) # Main Pipeline
pipeline_start = time.time()
news_df, num_gpt4 = add_sentiment(news_df, use_gpt4=USE_GPT4_FALLBACK)
pipeline_end = time.time()
print(f"Total pipeline runtime: {pipeline_end - pipeline_start:.2f} seconds.")

# Calculaitng the daily sentiment aggregate
sent_daily = (
    news_df.groupby(news_df["Date"].dt.normalize())["Sentiment"]
    .mean()
    .reset_index()
)
sent_daily_path = os.path.join(DERIVED_PATH, "daily_sentiment_aggregate.csv")
sent_daily.to_csv(sent_daily_path, index=False)

# Merging with market data for modeling (LSTM)
market_merge = pd.merge(
    market_df, 
    sent_daily, 
    left_on=market_df["Date"].dt.normalize(), 
    right_on=sent_daily["Date"].dt.normalize(), 
    how="left", 
    suffixes=('', '_sent')
)
final_out_path = os.path.join(DERIVED_PATH, "final_merged_for_lstm.csv")
market_merge.to_csv(final_out_path, index=False)
print(f"Checkpoint: Final merged LSTM-ready data saved to {final_out_path}")

print(f"All CSVs saved to: {DERIVED_PATH}/")
print("Output files:")
print(" - nyt_with_finbert_sentiment.csv")
print(" - nyt_with_gpt4_fallback_sentiment.csv")
print(" - daily_sentiment_aggregate.csv")
print(" - final_merged_for_lstm.csv")

finbert_only_path = os.path.join(DERIVED_PATH, "nyt_with_finbert_sentiment.csv") # Creating daily FinBERT-only sentiment aggregate for benchmarking
finbert_df = pd.read_csv(finbert_only_path, parse_dates=["Date"])

daily_finbert_sent = (
    finbert_df.groupby(finbert_df["Date"].dt.normalize())["FinBERT_score"]
    .mean()
    .reset_index()
)
daily_finbert_sent_path = os.path.join(DERIVED_PATH, "daily_FinBERT_sentiment_aggregate.csv")
daily_finbert_sent.to_csv(daily_finbert_sent_path, index=False)

# Merging with market data for modeling (LSTM) FinBERT-only version, acts as a benchmark
market_merge_finbert = pd.merge(
    market_df,
    daily_finbert_sent,
    left_on=market_df["Date"].dt.normalize(),
    right_on=daily_finbert_sent["Date"].dt.normalize(),
    how="left",
    suffixes=('', '_sent')
)
final_out_finbert_path = os.path.join(DERIVED_PATH, "final_merged_FinBERT_for_lstm.csv")
market_merge_finbert.to_csv(final_out_finbert_path, index=False)

print(f"Checkpoint: FinBERT-only daily sentiment aggregate saved to {daily_finbert_sent_path}")
print(f"Checkpoint: Final merged FinBERT-only LSTM-ready data saved to {final_out_finbert_path}")