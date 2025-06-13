# 📊 FAANMG 股票資料分析：資料與圖表來源說明

本專案使用 `yfinance` 下載美國六大科技股（FAANMG：META、AAPL、AMZN、NFLX、MSFT、GOOGL）之每日股價資料，並產出 K 線圖作為分析輔助。

---

## 📥 資料下載流程

以 Amazon (AMZN) 為例：

### 1️⃣ 安裝套件
請先於終端機安裝 `yfinance` 套件：

```bash
pip install yfinance
2️⃣ 下載資料並轉換格式（含 TimeIndex）
python
複製
編輯
import yfinance as yf
import pandas as pd

# 設定股票代碼與時間區間
ticker = "AMZN"
start_date = "2015-01-01"
end_date = "2025-01-01"

# 下載每日股價資料
data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# 還原 index 為欄位，並建立時間索引
data = data.reset_index()
data['TimeIndex'] = (data['Date'] - data['Date'].min()).dt.days

# 儲存成 CSV 檔
data.to_csv("AMZN_20150101_to_20250101.csv", index=False)
print("✅ 資料已儲存為 AMZN_20150101_to_20250101.csv")
📈 K 線圖產出流程
以 Amazon (AMZN) 為例：

1️⃣ 安裝套件
bash
複製
編輯
pip install yfinance mplfinance
2️⃣ 產生 K 線圖
python
複製
編輯
import yfinance as yf
import mplfinance as mpf

# 取得股價歷史資料
amzn = yf.Ticker("AMZN")
df = amzn.history(start="2015-01-01", end="2025-01-01")

# 繪製 K 線圖（含成交量）
if df.empty:
    print("❌ 資料下載失敗")
else:
    mpf.plot(df, type='candle', style='charles', title='AMZN', volume=True)
📁 輸出說明
AMZN_20150101_to_20250101.csv：Amazon 從 2015 年至 2025 年的日股價資料，含 TimeIndex 欄位

K 線圖：視覺化每日股價變化（Open, High, Low, Close）與成交量，用於報告或簡報展示

arduino
複製
編輯

如果你還要加入其他公司（AAPL、META 等），只需將程式中 `"AMZN"` 替換為其他代碼即可。  
