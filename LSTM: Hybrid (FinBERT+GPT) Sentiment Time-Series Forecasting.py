### Hybrid (FinBERT+GPT) Sentiment LSTM Time-Series Forecasting
### Implementation laid out by Jakob Aungiers https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

!pip install ipykernel
!pip install numpy pandas scikit-learn tensorflow keras matplotlib
!pip install --upgrade notebook
!pip install python-dotenv
!pip install joblib
!pip install scikit-learn
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install threadpoolctl
!pip install tensorflow
!pip install scikit-image
!pip install scikit-learn-intelex
!pip install keras

import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

data_path = "derived/final_merged_for_lstm.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])

df = df.drop(columns=[col for col in ['key_0', 'Date_sent'] if col in df.columns]) # Cleaning up columns

for col in df.columns: # Ensuring all columns except Date are numeric
    if col not in ['Date']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

# Feature engineering
exclude_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'] # Remove prices and volume
feature_cols = ['Return', 'VIX', 'Sentiment']

N_LAGS = 2 # Lagged features added below
for lag in range(1, N_LAGS+1):
    df[f"Return_lag{lag}"] = df["Return"].shift(lag)
    df[f"Sentiment_lag{lag}"] = df["Sentiment"].shift(lag)
feature_cols += [f"Return_lag{lag}" for lag in range(1, N_LAGS+1)]
feature_cols += [f"Sentiment_lag{lag}" for lag in range(1, N_LAGS+1)]
df = df.dropna().reset_index(drop=True)

# Train/val/test split by time
TRAIN_END = date(2021, 12, 31)
VAL_END = date(2022, 12, 31)
df["Date"] = pd.to_datetime(df["Date"])
df_train = df[df["Date"] <= pd.to_datetime(TRAIN_END)]
df_val = df[(df["Date"] > pd.to_datetime(TRAIN_END)) & (df["Date"] <= pd.to_datetime(VAL_END))]
df_test = df[df["Date"] > pd.to_datetime(VAL_END)]

# Scaling (IMPORTANT) 
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_val = scaler.transform(df_val[feature_cols])
X_test = scaler.transform(df_test[feature_cols])
y_train = df_train["Return"].values
y_val = df_val["Return"].values
y_test = df_test["Return"].values

# Creating LSTM Sequences
SEQ_LEN = 5
def create_sequences(x, y, seq_len):
    xs, ys = [], []
    for i in range(len(x) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LEN)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LEN)

# LSTM Model
model = Sequential([
    LSTM(50, input_shape=(SEQ_LEN, X_train_seq.shape[2])),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train
print("Training LSTM …")
model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=16,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stop],
    verbose=1,
)

print("Evaluating …") # Evaluation
pred_test = model.predict(X_test_seq).flatten()
rmse = np.sqrt(mean_squared_error(y_test_seq, pred_test))
print(f"Test RMSE: {rmse:.6f}")

actual_dir = (y_test_seq > 0) # Directional accuracy
pred_dir = (pred_test > 0)
acc = accuracy_score(actual_dir, pred_dir)
print(f"Directional accuracy: {acc:.2%}")

df_out = df_test.iloc[SEQ_LEN:].copy().reset_index(drop=True)
df_out["Predicted_Return"] = pred_test
df_out.to_csv("derived/lstm_test_predictions.csv", index=False)

print(df_out[["Date", "Return", "Predicted_Return"]].head())
print("Final LSTM model summary:")
model.summary()

# Diagnostics duplicates check
import pandas as pd

for name in ["derived/lstm_test_predictions.csv"]:
    df = pd.read_csv(name, parse_dates=["Date"])
    dup = df.duplicated(subset=["Date"]).sum()
    print(name, "rows:", len(df), "duplicates:", dup)
