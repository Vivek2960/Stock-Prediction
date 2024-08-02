from flask import Flask, render_template, request
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('lstm_stock_model.h5')

# Dictionary of companies and their stock tickers
COMPANIES = {
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'Google': 'GOOGL',
    'Microsoft': 'MSFT',
    'Tesla': 'TSLA',
    'Facebook': 'META',       # Meta Platforms (formerly Facebook)
    'Netflix': 'NFLX',
    'Nvidia': 'NVDA',
    'Alibaba': 'BABA',
    'Berkshire Hathaway': 'BRK-B',
    'Johnson & Johnson': 'JNJ',
    'JPMorgan Chase': 'JPM',
    'Visa': 'V',
    'Procter & Gamble': 'PG',
    'UnitedHealth Group': 'UNH',
    'Home Depot': 'HD',
    'MasterCard': 'MA',
    'Walt Disney': 'DIS',
    'Pfizer': 'PFE',
    'Coca-Cola': 'KO'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        company_name = request.form.get('company')
        ticker = COMPANIES[company_name]
        data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
        predictions, signal = predict_stock(data)
        plot_url = plot_predictions(data['Close'], predictions)
        return render_template('result.html', company=company_name, signal=signal, plot_url=plot_url)

    return render_template('index.html', companies=COMPANIES.keys())

def predict_stock(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    time_step = 60

    def create_sequences(data, time_step=60):
        X = []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
        return np.array(X)

    X_test = create_sequences(scaled_data)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    signal = "Profit" if predictions[-1] > data['Close'].iloc[-1] else "Loss"
    return predictions, signal

def plot_predictions(actual, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual, label='Actual Price')
    plt.plot(actual.index[-len(predictions):], predictions, label='Predicted Price', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
