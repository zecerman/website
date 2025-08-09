# TODO annotate
import io
import numpy as np 
import pandas as pd 
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import yfinance as yf
import torch
import torch.nn as nn
from datetime import datetime
import random as r
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer
from flask import Response

# Globals
device = 'cpu'
output = ''
is_complete = False
TICKER = 'UNASSIGNED'
# Variables that must be available to app.py to create plot
y_actuals = None
y_projection = None
y_preds_train = None
y_preds_test = None
extended_dates = None


# MAIN BEHAVIOR v
def run():
    global y_actuals, y_projection, y_preds_train, y_preds_test, extended_dates # Globals to be referenced by app.py
    global device, output, is_complete, TICKER # Internal globals
    is_complete = False

    try: 
        # Download and process data into loaders (record the date_series for later use in plotting)
        data, date_series, TICKER = data_step()
        train_loader, test_loader, input_size, target_scaler, test_dataset = init_loaders(data)

        # Init model
        model = MyLSTM(input_size=input_size).to(device)
        model.apply(init_weights)

        # Init training params
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
        NUM_EPOCHS = 20
        
        # Run training loop
        start_time = timer()
        model_results = train_loop(model=model,
                                train_dataloader=train_loader,
                                test_dataloader=test_loader,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                scheduler=scheduler,
                                epochs=NUM_EPOCHS)

        end_time = timer()
        
        # Model predictions are in logits, recover meaningfull values
        y_preds_train, y_actuals_train = inverse_scale(model, train_loader, scaler=target_scaler)
        y_preds_test, y_actuals_test = inverse_scale(model, test_loader, scaler=target_scaler)
        # Concat train/test vectrors
        y_preds = np.append(y_preds_train, y_preds_test)
        y_actuals = np.append(y_actuals_train, y_actuals_test)

        # Perform inference on simulated future data 
        # (synthetic data assumes ticker's final day performance continues)
        last_sequence = test_dataset[-1][0].unsqueeze(0).to(device)
        future_preds = forecast_future(model, last_sequence, n_steps=50, scaler=target_scaler,target_feature_idx=input_size-1)
        y_projection = np.append(y_preds, future_preds)

        # Extend date data
        last_date = date_series.iloc[-1] # Get the last date from the existing series
        extra_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_preds), freq='B')  # 'B' = business day
        extended_dates = pd.concat([date_series, pd.Series(extra_dates)]).reset_index(drop=True)

        # END OF IF BLOCK
        output = TICKER

    except Exception as e:
        output = f'Error: {str(e)}'

    # Cleanup
    print('hit') # NOT REACHED
    is_complete = True
    return output

def get_output():
    return output

def is_done():
    return is_complete
# MAIN BEHAVIOR ^

# Subfunctions of 'run()'
def data_step(): # TODO change ticker selection logic
    ticker_ls = [
        "AAPL",  # Apple Inc.
        "JPM",   # JPMorgan Chase & Co.
        "XOM",   # Exxon Mobil Corp.
        "CVX",   # Chevron Corp.
        #"WMT",   # Walmart Inc.
        "PG",    # Procter & Gamble
        "HD",    # Home Depot Inc.
        "DIS",   # The Walt Disney Company
        "NEE"    # NextEra Energy
    ]
    TICKER = ticker_ls[r.randint(0,len(ticker_ls)-1)]

    current_date = datetime.today()
    START_DATE = (current_date - relativedelta(years=5)).strftime('%Y-%m-%d')
    END_DATE = current_date.strftime('%Y-%m-%d')
    raw_data = yf.download(TICKER, start=START_DATE, end=END_DATE).reset_index()
    data = pd.DataFrame({
        'Date':raw_data[['Date']].values.ravel(),
        'Low':raw_data[['Low']].values.ravel(),
        'High':raw_data[['High']].values.ravel(),
        'Open':raw_data[['Open']].values.ravel(),
        'Close':raw_data[['Close']].values.ravel()
        })
    data.dropna(inplace=True)

    # Save datetime objects for later use as pretty printing
    date_series = data['Date']

    # Datetime objects incompatible w/ training, conv to ordinal integer representation
    data['Date'] = data['Date'].map(lambda x: x.toordinal())
    data['Date'] = (data['Date'] - min(data['Date']))

    # Add additional features which indicate price movement using pandas libraries
    data['SMA'] = ta.sma(data['Close'], length=10)
    data['EMA'] = ta.ema(data['Close'], length=20)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['ROC'] = ta.roc(data['Close'])
    data['RET'] = data['Close'].pct_change()

    # Extract target variable (if looking for final value of the ticker in one day)
    data['Target'] = data['Close'].shift(-1)
    data.drop(columns=['Close'], inplace=True)
    data.dropna(inplace=True)
    return data, date_series, TICKER

def init_loaders(data):
    # Num backcandles:
    N_backcandles = 90

    # Split raw data first (to avoid data leakage), unshuffled is appropriate for time series analysis 
    split_idx = int(len(data) * 0.85)
    data_train = data.iloc[:split_idx].copy()
    data_test = data.iloc[split_idx:].copy()

    # Separate features and target
    features_train = data_train.drop(columns=['Target']).values
    target_train = data_train['Target'].values.reshape(-1, 1)

    features_test = data_test.drop(columns=['Target']).values
    target_test = data_test['Target'].values.reshape(-1, 1)

    # Fit scalers only on training data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    scaled_features_train = feature_scaler.fit_transform(features_train)
    scaled_target_train = target_scaler.fit_transform(target_train)

    # Transform test data using fitted scalers
    scaled_features_test = feature_scaler.transform(features_test)
    scaled_target_test = target_scaler.transform(target_test)

    # Recombine scaled features and target
    scaled_data_train = np.concatenate((scaled_features_train, scaled_target_train), axis=1)
    scaled_data_test = np.concatenate((scaled_features_test, scaled_target_test), axis=1)

    # Make backcandles
    def generate_backcandles(data, target_col, n):
        X, y = [], []
        for i in range(n, len(data)):
            input_candle = np.delete(data[i - n:i], target_col, axis=1)  # shape: (n, num_features)
            X.append(input_candle)
            y.append(data[i, target_col])  # scalar target

        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        y = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32).to(device)
        return X, y

    # Generate train/test tensors
    target_col_index = scaled_data_train.shape[1] - 1  # same for both sets
    X_train, y_train = generate_backcandles(scaled_data_train, target_col=target_col_index, n=N_backcandles)
    X_test, y_test = generate_backcandles(scaled_data_test, target_col=target_col_index, n=N_backcandles)

    # Create TensorDatasets 
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Create Dataloaders
    BATCH_SIZE = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Record input size for later reference by other functions
    input_size = X_train.shape[2]
    # Return target_scaler for the same reason
    return train_loader, test_loader, input_size, target_scaler, test_dataset

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        # Architecture mostly relies the nn.LSTM() block
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Additional dense block 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)               # out: [batch_size, seq_len, hidden_size]
        out = self.norm(out[:, -1, :])      # Take only last time step
        out = self.dropout(out)
        return self.fc(out)                 # Final output: [batch_size, 1]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def inverse_scale(model, dataloader, scaler=None):
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    if scaler: 
        predictions = scaler.inverse_transform(predictions)
        actuals = scaler.inverse_transform(actuals)
    
    return predictions.flatten(), actuals.flatten()

def forecast_future(model, last_sequence, n_steps, scaler=None, date_feature_idx=0, target_feature_idx=101):
    model.eval()
    predictions = []

    seq = last_sequence.clone().detach().to(device)  # shape: [1, seq_len, num_features]

    # Get current max date from the last sequence
    current_date = seq[0, -1, date_feature_idx].item()

    for _ in range(n_steps):
        with torch.no_grad():
            pred = model(seq)  # shape: [1, 1]

        pred_value = pred.item()
        predictions.append(pred_value)

        # Use last feature vector as base
        last_features = seq[:, -1, :].clone()

        # Update the target and date
        last_features[0, target_feature_idx] = pred_value
        last_features[0, date_feature_idx] = current_date + 1  # increment date ordinal

        # Prepare next input step: shape [1, 1, num_features]
        next_input = last_features.unsqueeze(1)

        # Roll sequence window forward
        seq = torch.cat([seq[:, 1:, :], next_input], dim=1)

        # Update internal date tracker
        current_date += 1

    # Inverse scale the predictions if scaler is used
    if scaler:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    return predictions

def plot_predictions(actuals, predictions, dates, y_preds_train_len, y_preds_test_len):
    fig = Figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    shift = 4
    pred_len = len(predictions)
    x_pred = np.concatenate((np.array([0]*shift), np.array(range(pred_len - shift))))

    ax.plot(range(len(actuals)), actuals, label='Actual')
    ax.plot(x_pred, predictions, label='Predicted')
    ax.axvline(x=y_preds_train_len, color='black', linestyle='--', label='Train/Test Split')
    ax.axvline(x=y_preds_train_len + y_preds_test_len - 1, color='gray', linestyle='--', label='End of Dataset')

    tick_stride = max(len(dates) // 10, 1)
    tick_locs = list(range(0, len(dates), tick_stride))
    tick_labels = dates.iloc[tick_locs].dt.strftime('%Y-%m-%d')
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)

    #TODO includea toggle for this
    # Limit x-axis to right half
    half_point = len(actuals) // 1.5
    ax.set_xlim(half_point, len(actuals) + 10)

    ax.set_title(f"LSTM Model Stock Price Prediction for {TICKER}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price per Share (USD)")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)

    return Response(buf.getvalue(), mimetype='image/png')









# Train/test loop functions
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Accumulate loss
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss / total_samples

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # Accumulate weighted loss
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

    return total_loss / total_samples

def train_loop(model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            epochs: int):
    
    results = {
        'train loss' : [],
        'test loss' : []
    }
    # loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss = train_step(model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer)
        test_loss = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        scheduler.step(test_loss)

        results['train loss'].append(train_loss)
        results['test loss'].append(test_loss)
        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

    return results