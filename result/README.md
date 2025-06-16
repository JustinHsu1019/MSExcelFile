# 美股六大科技股分析結果

## 📊 分析結果概覽

本目錄包含對六大美股科技股（AAPL、AMZN、GOOGL、META、MSFT、NFLX）的完整預測分析結果。

## 📁 檔案結構

```
result/
├── Project_Results.csv           # 主要分析結果數據表
├── summary_comparison.png        # 總結對比圖表
├── AAPL_visualizations/         # Apple 詳細分析圖表
├── AMZN_visualizations/         # Amazon 詳細分析圖表
├── GOOGL_visualizations/        # Google 詳細分析圖表
├── META_visualizations/         # Meta 詳細分析圖表
├── MSFT_visualizations/         # Microsoft 詳細分析圖表
└── NFLX_visualizations/         # Netflix 詳細分析圖表
```

## 🎯 關鍵發現

- **最佳表現股票**：MSFT (MAE: 0.002796)
- **最具挑戰性**：NFLX (MAE: 0.004404)
- **最重要特徵**：MA5 (5日移動平均) - 所有股票共同的關鍵預測因子
- **平均預測區間覆蓋率**：85.1% (接近目標90%)

## 📈 主要圖表說明

- `summary_comparison.png`: 四合一總結圖，包含MAE比較、預算分佈、覆蓋率與MSIS分析
- 各股票資料夾內包含：
  - MAE分析圖
  - 預測準確度散點圖  
  - 時間序列預測對比
  - Top Features重要性分析

## 📋 數據格式

`Project_Results.csv` 包含以下欄位：
- `Ticker`: 股票代碼
- `CV-MAE_Uncon`: 無約束模型交叉驗證MAE
- `Best_Budget`: 最佳L1預算參數
- `CV-MAE_Best`: 最佳預算下的交叉驗證MAE
- `Coverage`: 預測區間覆蓋率
- `MSIS`: 平均標準化區間分數
- `Top_Feature`: 最重要預測特徵
- `Top_Importance`: 特徵重要性分數
