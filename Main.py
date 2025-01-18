import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error

stock = 'AAPL'
stock_data = yf.download(stock, start="2010-01-01", end="2023-01-01")

stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data.dropna(inplace=True)

X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = stock_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'stock_price_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')
