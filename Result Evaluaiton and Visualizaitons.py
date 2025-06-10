## Evaluaiton and Visualizaiton


### Forecast Biasness Test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

df_gpt   = pd.read_csv("derived/lstm_test_predictions.csv",
                       parse_dates=["Date"])
df_fbert = pd.read_csv("derived/lstm_FinBERT_only_test_predictions.csv",
                       parse_dates=["Date"])
market   = pd.read_csv("sp500_vix_data.csv", parse_dates=["Date"])

df = (df_gpt[["Date", "Return", "Predicted_Return"]]
        .rename(columns={"Return":"Market_Return",
                         "Predicted_Return":"LSTM_GPT4"})
      .merge(df_fbert[["Date","Predicted_Return"]]
                .rename(columns={"Predicted_Return":"LSTM_FinBERT"}),
             on="Date", how="left")
      .merge(market[["Date","Return","VIX"]]
                .rename(columns={"Return":"Market_True_Return"}),
             on="Date", how="left")
      .dropna(subset=["Market_True_Return","LSTM_GPT4","LSTM_FinBERT"]))

y_true   = df["Market_True_Return"].values # Handles
pred_gpt = df["LSTM_GPT4"].values
pred_fb  = df["LSTM_FinBERT"].values

def signal_metrics(y, yhat): # Forecast-Evaluation Metrics 
    """Return dict of point & trading metrics."""
    mse  = np.mean((y - yhat)**2)
    mae  = np.mean(np.abs(y - yhat))
    rmse = np.sqrt(mse)
    strat_ret = np.where(yhat>0, 1, -1) * y # Trading signal is long if yhat>0, short otherwise
    sharpe = np.mean(strat_ret)/(np.std(strat_ret)+1e-9)*np.sqrt(252)
    r2 = sm.OLS(y, sm.add_constant(yhat)).fit().rsquared
    return dict(MSE=mse, MAE=mae, Directional_Acc=acc,
                Sharpe=sharpe, R2=r2)
   
m_gpt = signal_metrics(y_true, pred_gpt)
m_fb  = signal_metrics(y_true, pred_fb)

def dm_test(e1, e2, h=1): # Diebold–Mariano (squared-error loss)
    d   = e1-e2
    T   = len(d)
    var = np.var(d, ddof=1) + 2*sum(
          np.cov(d[:-k],d[k:])[0,1] for k in range(1,h))
    dm  = np.mean(d)/np.sqrt(var/T)
    p   = 2*(1-stats.norm.cdf(abs(dm)))
    return dm, p

dm_stat, dm_p = dm_test((y_true-pred_gpt)**2, (y_true-pred_fb)**2)

df["VIX"] = pd.to_numeric(df["VIX"], errors="coerce") # Regime (VIX median) Sharpe comparison
vix_med = df["VIX"].median()
reg   = {}
for regime, lab in [(df["VIX"]>vix_med,"High-VIX"),
                    (df["VIX"]<=vix_med,"Low-VIX")]:
    reg[lab] = dict(GPT4   = signal_metrics(
                                y_true[regime], pred_gpt[regime])["Sharpe"],
                    FinBERT = signal_metrics(
                                y_true[regime], pred_fb[regime])["Sharpe"])

def bias_tests(y, yhat, label): # Forecast-Bias Section
    err = y - yhat

    t, p = stats.ttest_1samp(err, 0.0) # Mean-error t-test

    X = sm.add_constant(yhat) # Mincer–Zarnowitz regression
    mz = sm.OLS(y, X).fit()

    # Joint test α=0, β=1  (forecast unbiased)
    R = np.eye(2)
    q = np.array([0,1])
    ftest = mz.f_test((R,q))
    return dict(
        Model          = label,
        Mean_Error     = err.mean(),
        MeanErr_tstat  = t,
        MeanErr_pval   = p,
        MZ_alpha       = mz.params[0],
        MZ_beta        = mz.params[1],
        MZ_alpha_p     = mz.pvalues[0],
        MZ_beta_p      = mz.pvalues[1],
        MZ_F_pvalue    = float(ftest.pvalue)
    )

bias_gpt = bias_tests(y_true, pred_gpt, "Hybrid (GPT-4)")
bias_fb  = bias_tests(y_true, pred_fb , "FinBERT-only")

from tabulate import tabulate

# Forecast-evaluation
eval_tbl = pd.DataFrame([m_gpt, m_fb], index=["Hybrid","FinBERT"])
print("Forecast-Evaluation Metrics")
print(tabulate(eval_tbl, headers="keys", tablefmt="github", floatfmt=".4f"))

print("\nDiebold–Mariano statistic: {:.3f}   p-value: {:.3f}" # DM-test summary
      .format(dm_stat, dm_p))

print("Regime-dependent Sharpe") # Regime Sharpe
print(tabulate(pd.DataFrame(reg).T, headers="keys",
               tablefmt="github", floatfmt=".3f"))

bias_tbl = pd.DataFrame([bias_gpt, bias_fb]).set_index("Model") # Bias tests
print("Forecast-Bias Diagnostics")
print(tabulate(bias_tbl, headers="keys", tablefmt="github", floatfmt=".4f"))

from sklearn.metrics import accuracy_score # computing validation accuracy for both models

print("Validation Accuracy")

val_acc_gpt = accuracy_score(
    (df["Market_True_Return"].values > 0),   # true direction
    (df["LSTM_GPT4"].values          > 0)    # predicted direction
)

val_acc_fb = accuracy_score(
    (df["Market_True_Return"].values > 0),
    (df["LSTM_FinBERT"].values       > 0)
)

print(f"Hybrid (GPT-4) Validation Accuracy:  {val_acc_gpt:.2%}")
print(f"FinBERT-only Validation Accuracy:    {val_acc_fb:.2%}")


### Evaluation Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from scipy.stats import norm

df_gpt = pd.read_csv("derived/lstm_test_predictions.csv", parse_dates=["Date"])
df_fbert = pd.read_csv("derived/lstm_FinBERT_only_test_predictions.csv", parse_dates=["Date"])
market_df = pd.read_csv("sp500_vix_data.csv", parse_dates=["Date"])

df = pd.DataFrame({ # Aligning data by date
    "Date": df_gpt["Date"],
    "Market_Return": df_gpt["Return"],
    "LSTM_GPT4": df_gpt["Predicted_Return"],
    "LSTM_FinBERT": df_fbert["Predicted_Return"],
})
df = df.merge(market_df[["Date", "Return", "VIX"]].rename(columns={"Return": "Market_True_Return"}), on="Date", how="left")
df["VIX"] = pd.to_numeric(df["VIX"], errors="coerce")

def trading_signal_returns(true_returns, predicted_returns): # Signal-based trading returns
    signal = np.where(predicted_returns > 0, 1, -1)
    return signal * true_returns

def compute_all_metrics(true_returns, predicted_returns, rolling_window=60): # Computing metrics for each model (with rolling mean/std)
    if len(true_returns) == 0 or len(predicted_returns) == 0:
        raise ValueError("Empty input arrays! Check your mask and input data.")
    mse = mean_squared_error(true_returns, predicted_returns)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_returns, predicted_returns)

    true_up = (true_returns > 0).astype(int)
    pred_up = (predicted_returns > 0).astype(int)

    acc = accuracy_score(true_up, pred_up)
    prec = precision_score(true_up, pred_up, zero_division=0)
    rec = recall_score(true_up, pred_up, zero_division=0)
    f1 = f1_score(true_up, pred_up, zero_division=0)
    try:
        roc = roc_auc_score(true_up, predicted_returns)
    except:
        roc = np.nan

    cm = confusion_matrix(true_up, pred_up)
    strat_returns = trading_signal_returns(true_returns, predicted_returns)
    cum_return = np.cumprod(1 + strat_returns)[-1] - 1 if len(strat_returns) > 0 else np.nan
    sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-9) * np.sqrt(252)
    roll_sharpe = pd.Series(strat_returns).rolling(rolling_window).apply(
        lambda x: np.mean(x) / (np.std(x) + 1e-9) * np.sqrt(252), raw=True)
    roll_acc = pd.Series(pred_up == true_up).rolling(rolling_window).mean()
    roll_cum_return = (1 + pd.Series(strat_returns)).cumprod() - 1
    rolling_sharpe_mean, rolling_sharpe_std = roll_sharpe.mean(), roll_sharpe.std()
    rolling_acc_mean, rolling_acc_std = roll_acc.mean(), roll_acc.std()
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae,
        "Direction_Acc": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc,
        "Sharpe": sharpe, "Cumulative_Return": cum_return,
        "Rolling_Sharpe_Mean": rolling_sharpe_mean, "Rolling_Sharpe_Std": rolling_sharpe_std,
        "Rolling_Acc_Mean": rolling_acc_mean, "Rolling_Acc_Std": rolling_acc_std,
        "Confusion_Matrix": cm,
        "Rolling_Sharpe": roll_sharpe,
        "Rolling_Acc": roll_acc,
        "Rolling_CumReturn": roll_cum_return,
        "Signal_Returns": strat_returns
    }

mask_gpt = df['Market_True_Return'].notnull() & df['LSTM_GPT4'].notnull() # Creating valid data masks and calculate for both models
mask_fbert = df['Market_True_Return'].notnull() & df['LSTM_FinBERT'].notnull()

print("\nValid rows for GPT:", mask_gpt.sum())
print("Valid rows for FinBERT:", mask_fbert.sum())

if mask_gpt.sum() == 0 or mask_fbert.sum() == 0:
    raise ValueError("No valid data rows for at least one model. Check input data and merges!")

metrics_gpt = compute_all_metrics(
    df.loc[mask_gpt, 'Market_True_Return'].values,
    df.loc[mask_gpt, 'LSTM_GPT4'].values
)
metrics_finbert = compute_all_metrics(
    df.loc[mask_fbert, 'Market_True_Return'].values,
    df.loc[mask_fbert, 'LSTM_FinBERT'].values
)

# Table (include rolling mean/std)
comparison_table = pd.DataFrame({
    "Hybrid (FinBERT+GPT-4)": {k: v for k, v in metrics_gpt.items() if not isinstance(v, (np.ndarray, pd.Series, list))},
    "FinBERT-only": {k: v for k, v in metrics_finbert.items() if not isinstance(v, (np.ndarray, pd.Series, list))}
})
print("\n===== Model Comparison Table =====\n")
print(comparison_table)
comparison_table.to_csv("model_performance_comparison.csv")

# 4. Regime split (by VIX): show model dominance in high/low volatility
vix_median = df["VIX"].median()
df["VIX_regime"] = np.where(df["VIX"] > vix_median, "High_VIX", "Low_VIX")
def regime_metrics(regime):
    idx = df["VIX_regime"] == regime
    gpt_metrics = compute_all_metrics(
        df.loc[idx & mask_gpt, "Market_True_Return"].values, df.loc[idx & mask_gpt, "LSTM_GPT4"].values
    )
    finbert_metrics = compute_all_metrics(
        df.loc[idx & mask_fbert, "Market_True_Return"].values, df.loc[idx & mask_fbert, "LSTM_FinBERT"].values
    )
    return gpt_metrics, finbert_metrics
for regime in ["High_VIX", "Low_VIX"]:
    gpt_metrics, finbert_metrics = regime_metrics(regime)
    print(f"\n=== {regime} Regime ===")
    print("Hybrid Sharpe:", gpt_metrics["Sharpe"], "Cumulative:", gpt_metrics["Cumulative_Return"])
    print("FinBERT-only Sharpe:", finbert_metrics["Sharpe"], "Cumulative:", finbert_metrics["Cumulative_Return"])


### Resulting Plots
import matplotlib.pyplot as plt
print(plt.style.available)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, r2_score
)

fbert = pd.read_csv("derived/lstm_FinBERT_only_test_predictions.csv", parse_dates=["Date"])
gpt4 = pd.read_csv("derived/lstm_test_predictions.csv", parse_dates=["Date"])
market = pd.read_csv("sp500_vix_data.csv", parse_dates=["Date"])

for df in [fbert, gpt4]: # changing objects (str) columns to numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

plt.style.use("seaborn-v0_8-darkgrid") # General settings
sns.set(font_scale=1.2)
window = 21  # 1 month rolling was chosen

df = fbert[["Date", "Return", "VIX", "FinBERT_score", "Predicted_Return"]].copy() # Mergeing for side-by-side plots
df = df.rename(columns={"Predicted_Return": "FinBERT_Pred"})
df["GPT4_Pred"] = gpt4["Predicted_Return"].values
df["GPT4_Sentiment"] = gpt4["Sentiment"].values if "Sentiment" in gpt4 else np.nan


# Actual vs LSTM Forecasts: Time Series Overlap
plt.figure(figsize=(17,7))
plt.plot(df["Date"], df["Return"], color='black', label='Market Return', linewidth=2)
plt.plot(df["Date"], df["FinBERT_Pred"], color='orange', label='FinBERT-only LSTM', alpha=0.8)
plt.plot(df["Date"], df["GPT4_Pred"], color='red', label='FinBERT+GPT-4 LSTM', alpha=0.7)
plt.ylabel("Return")
plt.title("Market Returns vs LSTM Model Forecasts")
plt.legend()
plt.show()


# Model Error: Residuals Over Time 
plt.figure(figsize=(17,7))
plt.plot(df["Date"], df["Return"] - df["FinBERT_Pred"], label="Error: FinBERT LSTM", color="orange", alpha=0.6)
plt.plot(df["Date"], df["Return"] - df["GPT4_Pred"], label="Error: GPT-4 LSTM", color="dodgerblue", alpha=0.6)
plt.axhline(0, color="black", linewidth=1, linestyle=":")
plt.ylabel("Residual (Error)")
plt.title("Model Residuals: Market - Model Forecast")
plt.legend()
plt.show()


# Distribution of Forecasts: Histogram & KDE 
plt.figure(figsize=(14,5))
sns.histplot(df["Return"], label="Market Return", color="black", kde=True, stat="density", bins=40)
sns.histplot(df["FinBERT_Pred"], label="FinBERT LSTM", color="orange", kde=True, stat="density", bins=40, alpha=0.5)
sns.histplot(df["GPT4_Pred"], label="GPT-4 LSTM", color="dodgerblue", kde=True, stat="density", bins=40, alpha=0.5)
plt.legend()
plt.title("Distribution of Market Returns vs LSTM Model Forecasts")
plt.show()


# Actual vs Predicted: Scatter Plots and Regression Fit
fig, axs = plt.subplots(1, 2, figsize=(16,6), sharey=True)
sns.regplot(x=df["Return"], y=df["FinBERT_Pred"], ax=axs[0], line_kws={"color": "orange"})
axs[0].set_title("FinBERT LSTM: Actual vs Predicted")
axs[0].set_xlabel("Actual Return")
axs[0].set_ylabel("Predicted Return")
sns.regplot(x=df["Return"], y=df["GPT4_Pred"], ax=axs[1], line_kws={"color": "dodgerblue"})
axs[1].set_title("GPT-4 LSTM: Actual vs Predicted")
axs[1].set_xlabel("Actual Return")
plt.show()


# Rolling Model RMSE (21d window)
rmse_fbert = (df["Return"] - df["FinBERT_Pred"]).rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
rmse_gpt4 = (df["Return"] - df["GPT4_Pred"]).rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
plt.figure(figsize=(17,6))
plt.plot(df["Date"], rmse_fbert, label="Rolling RMSE: FinBERT LSTM", color="orange")
plt.plot(df["Date"], rmse_gpt4, label="Rolling RMSE: GPT-4 LSTM", color="dodgerblue")
plt.ylabel("RMSE")
plt.title("21-Day Rolling RMSE: Model Performance Over Time")
plt.legend()
plt.show()


# ROC Curve: Directional Signal of Models
from sklearn.metrics import roc_curve, auc
true_bin = (df["Return"] > 0).astype(int)
fpr_fbert, tpr_fbert, _ = roc_curve(true_bin, df["FinBERT_Pred"])
fpr_gpt4, tpr_gpt4, _ = roc_curve(true_bin, df["GPT4_Pred"])
auc_fbert = auc(fpr_fbert, tpr_fbert)
auc_gpt4 = auc(fpr_gpt4, tpr_gpt4)
plt.figure(figsize=(8,6))
plt.plot(fpr_fbert, tpr_fbert, label=f"FinBERT LSTM (AUC={auc_fbert:.2f})", color="orange")
plt.plot(fpr_gpt4, tpr_gpt4, label=f"GPT-4 LSTM (AUC={auc_gpt4:.2f})", color="dodgerblue")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Model Directional Prediction Accuracy")
plt.legend()
plt.show()


# Error Autocorrelation
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(8,4))
autocorrelation_plot(df["Return"] - df["FinBERT_Pred"])
plt.title("Error Autocorrelation: FinBERT LSTM")
plt.show()
plt.figure(figsize=(8,4))
autocorrelation_plot(df["Return"] - df["GPT4_Pred"])
plt.title("Error Autocorrelation: GPT-4 LSTM")
plt.show()


# Cumulative Return Plot
df = pd.DataFrame({ # Aligning data by date
    "Date": df_gpt["Date"],
    "Market_Return": df_gpt["Return"],  # Use market_df["Return"] if better aligned
    "LSTM_GPT4": df_gpt["Predicted_Return"],
    "LSTM_FinBERT": df_fbert["Predicted_Return"],
})
df = df.merge(market_df[["Date", "Return", "VIX"]].rename(columns={"Return": "Market_True_Return"}), on="Date", how="left")
df["VIX"] = pd.to_numeric(df["VIX"], errors="coerce")

def trading_signal_returns(true_returns, predicted_returns): # Signal-based trading returns
    signal = np.where(predicted_returns > 0, 1, -1)  # Long/short signal
    return signal * true_returns

mask_gpt = df['Market_True_Return'].notnull() & df['LSTM_GPT4'].notnull() # Establishing valid data masks and calculate for both models
mask_fbert = df['Market_True_Return'].notnull() & df['LSTM_FinBERT'].notnull()

print("\nValid rows for GPT:", mask_gpt.sum())
print("Valid rows for FinBERT:", mask_fbert.sum())

if mask_gpt.sum() == 0 or mask_fbert.sum() == 0:
    raise ValueError("No valid data rows for at least one model. Check input data and merges!")

metrics_gpt = compute_all_metrics(
    df.loc[mask_gpt, 'Market_True_Return'].values,
    df.loc[mask_gpt, 'LSTM_GPT4'].values
)
metrics_finbert = compute_all_metrics(
    df.loc[mask_fbert, 'Market_True_Return'].values,
    df.loc[mask_fbert, 'LSTM_FinBERT'].values
)

plt.figure(figsize=(12,5))

dates = df.loc[mask_gpt, 'Date'] # Align dates

gpt_signal = np.where(df.loc[mask_gpt, 'LSTM_GPT4'] > 0, 1, -1) # Computing trading strategy returns for both models
finbert_signal = np.where(df.loc[mask_fbert, 'LSTM_FinBERT'] > 0, 1, -1)

gpt_strat_returns = gpt_signal * df.loc[mask_gpt, 'Market_True_Return'].values
finbert_strat_returns = finbert_signal * df.loc[mask_fbert, 'Market_True_Return'].values

gpt_cum_return = np.cumsum(gpt_strat_returns) # Computing cumulative returns
finbert_cum_return = np.cumsum(finbert_strat_returns)
market_cum_return = np.cumsum(df.loc[mask_gpt, 'Market_True_Return'].values)

dates = df.loc[mask_gpt, 'Date'].reset_index(drop=True)

plt.figure(figsize=(12, 5))
plt.plot(dates, market_cum_return, label="Market Cumulative Return", color='black', linewidth=2)
plt.plot(dates, gpt_cum_return, label="Hybrid Cumulative Return", color='red')
plt.plot(dates, finbert_cum_return, label="FinBERT-only Cumulative Return", color='darkorange')
plt.title("Cumulative Strategy Returns Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()


# Rolling Directional Accuracy and Sharpe Ratio 
win = 63      
def rolling_directional_acc(true_ret, pred_ret, win=win):
    """
    Percentage of days inside a rolling window where sign(pred)==sign(true).
    win=63 ≈ one quarter of trading days.
    """
    hit = (np.sign(true_ret) == np.sign(pred_ret)).astype(int)
    return pd.Series(hit).rolling(win, min_periods=1).mean() * 100  # % scale

mkt_true = df.loc[mask_gpt, "Market_True_Return"].values  # Aligning the same true-return vector you used for Sharpe
pred_gpt = df.loc[mask_gpt, "LSTM_GPT4"].values
pred_fbt = df.loc[mask_fbert, "LSTM_FinBERT"].values
dates_da = df.loc[mask_gpt, "Date"].values

da_gpt   = rolling_directional_acc(mkt_true, pred_gpt)
da_fbt   = rolling_directional_acc(mkt_true, pred_fbt)

dates_all = df.loc[mask_gpt, "Date"].values 

def rolling_sharpe(returns, win=win): # Defining sharpe_gpt and sharpe_fbert
    """
    Calculate rolling Sharpe ratio over a specified window.
    Returns annualised Sharpe ratio.
    """
    mean_ret = returns.rolling(win).mean()
    std_ret  = returns.rolling(win).std(ddof=0)  # population std
    return (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)  # annualised
sharpe_gpt   = rolling_sharpe(df.loc[mask_gpt, "Market_True_Return"] - df.loc[mask_gpt, "LSTM_GPT4"])
sharpe_fbert = rolling_sharpe(df.loc[mask_fbert, "Market_True_Return"] - df.loc[mask_fbert, "LSTM_FinBERT"])

trim = win - 1 # Trimming FIRST (win-1) observations where rolling metrics
dates_trim        = dates_all[trim:]          # shared for both plots
sharpe_gpt_trim   = sharpe_gpt.iloc[trim:]
sharpe_fbt_trim   = sharpe_fbert.iloc[trim:]
da_gpt_trim       = da_gpt.iloc[trim:]
da_fbt_trim       = da_fbt.iloc[trim:]

plt.figure(figsize=(15,5)) # Rolling Sharpe (after trim)
plt.plot(dates_trim, sharpe_gpt_trim,   label="Hybrid rolling Sharpe",  c="red")
plt.plot(dates_trim, sharpe_fbt_trim,   label="FinBERT rolling Sharpe", c="darkorange")
plt.axhline(0, ls='--', c='k')
plt.title(f"Rolling Sharpe Ratio ({win}-day Window)")
plt.ylabel("Sharpe Ratio (annualised)")
plt.xlabel("Date")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(15,5)) # Rolling Directional-Accuracy (after trim)
plt.plot(dates_trim, da_gpt_trim, label="Hybrid directional accuracy",  c="red")
plt.plot(dates_trim, da_fbt_trim, label="FinBERT directional accuracy", c="darkorange")
plt.axhline(50, ls='--', c='k', lw=0.8)       # coin-flip baseline
plt.ylim(0, 100)
plt.title(f"Rolling Directional Accuracy ({win}-day Window)")
plt.ylabel("Accuracy (%)")
plt.xlabel("Date")
plt.legend(); plt.tight_layout(); plt.show()


# Confusion Matrix
from sklearn.metrics import confusion_matrix

def plot_conf_mat(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true > 0, y_pred > 0)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], ax=ax)
    ax.set_xlabel("Model Signal"); ax.set_ylabel("Market Direction"); ax.set_title(title)

fig, axes = plt.subplots(1,2,figsize=(10,4))
plot_conf_mat(df.loc[mask_gpt,"Market_True_Return"],
              df.loc[mask_gpt,"LSTM_GPT4"],
              "Hybrid", axes[0])
plot_conf_mat(df.loc[mask_fbert,"Market_True_Return"],
              df.loc[mask_fbert,"LSTM_FinBERT"],
              "FinBERT-only", axes[1])
plt.tight_layout(); plt.show()


# Turnover Function Vs Net-Long Ratio
def turnover(signal):
    return (np.abs(np.diff(signal)) / 2).mean() # Average fraction of days where direction flips

sig_gpt   = np.where(df.loc[mask_gpt,  'LSTM_GPT4']   > 0, 1, -1)
sig_fbert = np.where(df.loc[mask_fbert,'LSTM_FinBERT']> 0, 1, -1)

print(f"Hybrid turnover:  {turnover(sig_gpt):.2%}")
print(f"FinBERT turnover: {turnover(sig_fbert):.2%}")
print(f"Hybrid net-long ratio:  {(sig_gpt==1).mean():.2%}")
print(f"FinBERT net-long ratio: {(sig_fbert==1).mean():.2%}")

import matplotlib.pyplot as plt
import numpy as np

sig_gpt   = np.where(df.loc[mask_gpt,  'LSTM_GPT4']   > 0, 1, -1) # Re-computing signals (or reuse sig_gpt / sig_fbert from earlier)
sig_fbert = np.where(df.loc[mask_fbert,'LSTM_FinBERT']> 0, 1, -1)

def turnover(s):                      # share of days that flip direction
    return (np.abs(np.diff(s)) / 2).mean()

turn = [turnover(sig_gpt), turnover(sig_fbert)]
netL = [(sig_gpt == 1).mean(), (sig_fbert == 1).mean()]   # net-long ratios

labels = ["Hybrid\n(FinBERT+GPT-4)", "FinBERT-only"]
x = np.arange(len(labels))

fig, ax1 = plt.subplots(figsize=(8, 5))
width = 0.35

bars1 = ax1.bar(x - width/2, turn, width,    # Left y-axis => Turnover
                color="steelblue", alpha=.85, label="Turnover")
ax1.set_ylabel("Turnover (% of days)", color="steelblue")
ax1.set_ylim(0, 1)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.tick_params(axis='y', labelcolor="steelblue")

ax2 = ax1.twinx()   # Right y-axis => Net-Long exposure
bars2 = ax2.bar(x + width/2, netL, width,
                color="darkorange", alpha=.85, label="Net-Long Ratio")
ax2.set_ylabel("Net-Long Exposure", color="darkorange")
ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', labelcolor="darkorange")

for rect in bars1 + bars2:   # Annotating bars
    height = rect.get_height()
    ax = ax1 if rect in bars1 else ax2
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.02,
            f"{height:.2%}", ha='center', va='bottom', fontsize=9)

plt.title("Turnover vs Net-Long Ratio", pad=18)
plt.tight_layout()
plt.show()