### FinBERT-only Sentiment LSTM Time-Series Forecasting
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
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

data_path = "derived/final_merged_FinBERT_for_lstm.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
 
df = df.drop(columns=[col for col in ['key_0', 'Date_sent'] if col in df.columns]) # Cleaning columns

for col in ['Close', 'High', 'Low', 'Open', 'Volume']: # Removing column headers like '^GSPC' in Close/Open/etc (non-numeric entries)
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if "VIX" in df.columns:
    df["VIX"] = pd.to_numeric(df["VIX"], errors='coerce')

for col in df.columns: # Ensure all except 'Date' are numeric
    if col != 'Date':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)

# Feature Engineering 
N_LAGS = 2  # Adding lags for target and sentiment
for lag in range(1, N_LAGS+1):
    df[f"Return_lag{lag}"] = df["Return"].shift(lag)
    df[f"FinBERT_score_lag{lag}"] = df["FinBERT_score"].shift(lag)
df = df.dropna().reset_index(drop=True)

# Train/val/test split 
TRAIN_END = date(2021, 12, 31)
VAL_END = date(2022, 12, 31)

df["Date"] = pd.to_datetime(df["Date"])
df_train = df[df["Date"] <= pd.to_datetime(TRAIN_END)]
df_val = df[(df["Date"] > pd.to_datetime(TRAIN_END)) & (df["Date"] <= pd.to_datetime(VAL_END))]
df_test = df[df["Date"] > pd.to_datetime(VAL_END)]

# Features and Scaling
FEATURES = ['Close', 'High', 'Low', 'Open', 'Volume', 'VIX', 'FinBERT_score'] + \
           [f"Return_lag{lag}" for lag in range(1, N_LAGS+1)] + \
           [f"FinBERT_score_lag{lag}" for lag in range(1, N_LAGS+1)]
X_train = df_train[FEATURES].astype(np.float32).values
X_val = df_val[FEATURES].astype(np.float32).values
X_test = df_test[FEATURES].astype(np.float32).values
y_train = df_train["Return"].values
y_val = df_val["Return"].values
y_test = df_test["Return"].values

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Creating LSTM sequences (window=5)
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

print(f"Train samples: {X_train_seq.shape[0]}, Val: {X_val_seq.shape[0]}, Test: {X_test_seq.shape[0]}")

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

df_test = df_test.iloc[SEQ_LEN:].copy()  # aligning with y_test_seq
df_test["Predicted_Return"] = pred_test
df_test.to_csv("derived/lstm_FinBERT_only_test_predictions.csv", index=False)

print(df_test[["Date", "Return", "Predicted_Return"]].head())
print("Final LSTM model summary:")
model.summary()

# Diagnostics duplicates check
import pandas as pd

for name in ["derived/lstm_FinBERT_only_test_predictions.csv"]:
    df = pd.read_csv(name, parse_dates=["Date"])
    dup = df.duplicated(subset=["Date"]).sum()
    print(name, "rows:", len(df), "duplicates:", dup)
