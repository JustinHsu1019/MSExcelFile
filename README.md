取資料的過程
1.安裝 yfinance 套件：
pip install yfinance 
2.然後以 python 執行以下程式（以 Amazon 為例）：
import yfinance as yf
import pandas as pd
ticker = "AMZN"
start_date = "2015-01-01"
end_date = "2025-01-01"
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
data = data.reset_index() 
data['TimeIndex'] = (data['Date'] - data['Date'].min()).dt.days
data.to_csv("AMZN_20150101_to_20250101.csv", index=False)
print("AMZN_20150101_to_20250101.csv")

取圖片的過程
1.安裝 yfinance 及 mplfinance套件：
pip install yfinance mplfinance
2.然後以 python 執行以下程式（以 Amazon 為例）：
import yfinance as yf
import mplfinance as mpf
amzn = yf.Ticker("AMZN")
df = amzn.history(start="2015-01-01", end="2025-01-01")
if df.empty:
   print("FAIL")
else:
   mpf.plot(df, type='candle', style='charles', title='AMZN', volume=True)
