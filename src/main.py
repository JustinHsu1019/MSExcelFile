import yfinance as yf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from gurobipy import Model, GRB, quicksum
from scipy.stats import laplace
import statsmodels.api as sm

# 環境設置
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
KNOT_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
BUDGET_LEVELS = [0.5, 1, 2, 5, 10, 20, 50]  # 預算範圍
SAVE_DIR = "result"
os.makedirs(SAVE_DIR, exist_ok=True)

# 設置matplotlib中文字體和風格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. 資料準備
def prepare_data(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data.reset_index()
    
    # 核心預測變量
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Return_lag1'] = data['LogReturn'].shift(1)
    data['Return_lag2'] = data['LogReturn'].shift(2)
    data['LogVolume'] = np.log(data['Volume'] + 1e-6)
    data['LogVolume_lag1'] = data['LogVolume'].shift(1)
    data['Volatility'] = data['LogReturn'].rolling(5).std()
    data['RSI'] = compute_rsi(data['Close'], 14)  # 相對強弱指數
    
    # 技術指標
    data['MA5'] = data['Close'].rolling(5).mean() / data['Close'] - 1
    data['MA20'] = data['Close'].rolling(20).mean() / data['Close'] - 1

    # 日期特徵
    data['TimeIndex'] = (data['Date'] - data['Date'].min()).dt.days
    
    return data.dropna().reset_index(drop=True)

# 2. RSI計算函數
def compute_rsi(prices, window):
    deltas = prices.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    
    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

# 3. 按預測變量分組創建knots
def create_knots_by_predictor(train_data, predictors):
    knots_dict = {}
    for predictor in predictors:
        knots_dict[predictor] = [
            np.percentile(train_data[predictor], q * 100) 
            for q in KNOT_PERCENTILES
        ]
    return knots_dict

# 4. 為所有預測變量創建hinge特徵
def add_all_hinge_features(df, knots_dict):
    df_copy = df.copy()
    for predictor, knots in knots_dict.items():
        for knot in knots:
            hinge_col = f"H_{predictor}_{knot:.4f}"
            df_copy[hinge_col] = np.maximum(0, df_copy[predictor] - knot)
    return df_copy

# 5. 完整樣條回歸模型
def fit_spline_l1_model(X_train, y_train, budget=None):
    n, p = X_train.shape
    model = Model()
    model.setParam("OutputFlag", 0)

    beta0 = model.addVar(lb=-GRB.INFINITY, name="intercept")
    coefs = model.addVars(p, lb=-GRB.INFINITY, name="beta")
    pos_errors = model.addVars(n, lb=0, name="pos_err")
    neg_errors = model.addVars(n, lb=0, name="neg_err")
    
    # L1正則化約束
    if budget is not None:
        abs_coefs = model.addVars(p, lb=0, name="abs_beta")
        for j in range(p):
            model.addConstr(coefs[j] <= abs_coefs[j])
            model.addConstr(-coefs[j] <= abs_coefs[j])
        model.addConstr(quicksum(abs_coefs[j] for j in range(p)) <= budget)
    
    # 添加擬合約束
    for i in range(n):
        pred = beta0 + quicksum(coefs[j] * X_train[i, j] for j in range(p))
        model.addConstr(y_train[i] - pred <= pos_errors[i])
        model.addConstr(pred - y_train[i] <= neg_errors[i])
        model.addConstr(pos_errors[i] + neg_errors[i] >= 0)  # 誤差非負
    
    # 最小化絕對誤差
    total_error = quicksum(pos_errors[i] + neg_errors[i] for i in range(n))
    model.setObjective(total_error, GRB.MINIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        beta_vals = np.array([coefs[j].X for j in range(p)])
        return beta0.X, beta_vals, model.ObjVal
    else:
        return None, None, None

# 6. Part I
def part_i_unconstrained(data, predictors):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_values = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        # 創建knots字典（按預測變量）
        knots_dict = create_knots_by_predictor(train, predictors)
        
        # 添加hinge特徵
        train_hinge = add_all_hinge_features(train, knots_dict)
        test_hinge = add_all_hinge_features(test, knots_dict)
        
        # 識別特徵列
        base_features = predictors.copy()
        all_columns = list(train_hinge.columns)
        hinge_features = [col for col in all_columns if col.startswith('H_')]
        all_features = base_features + hinge_features
        
        # 準備資料
        X_train = train_hinge[all_features].values
        y_train = train_hinge['LogReturn'].values
        X_test = test_hinge[all_features].values
        y_test = test_hinge['LogReturn'].values
        
        # 擬合無約束模型
        beta0, beta, _ = fit_spline_l1_model(X_train, y_train)
        
        if beta0 is not None:
            y_pred = beta0 + X_test @ beta
            mae = np.mean(np.abs(y_test - y_pred))
            mae_values.append(mae)
            print(f"Fold {fold+1}: MAE = {mae:.6f}")
    
    return np.mean(mae_values) if mae_values else np.nan, knots_dict

# 7. Part II: 預算約束優化
def part_ii_budgeted(data, predictors, knots_dict, budget_levels=BUDGET_LEVELS):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    coef_analysis = {B: [] for B in budget_levels}
    detailed_results = {B: {'fold_maes': [], 'predictions': [], 'actuals': []} for B in budget_levels}
    
    for B in budget_levels:
        mae_values = []
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            
            # 添加hinge特徵（使用全局knots）
            train_hinge = add_all_hinge_features(train, knots_dict)
            test_hinge = add_all_hinge_features(test, knots_dict)
            
            # 識別特徵列
            base_features = predictors.copy()
            all_columns = list(train_hinge.columns)
            hinge_features = [col for col in all_columns if col.startswith('H_')]
            all_features = base_features + hinge_features
            
            # 準備資料
            X_train = train_hinge[all_features].values
            y_train = train_hinge['LogReturn'].values
            X_test = test_hinge[all_features].values
            y_test = test_hinge['LogReturn'].values
            
            # 擬合約束模型
            beta0, beta, obj_val = fit_spline_l1_model(X_train, y_train, budget=B)
            
            if beta0 is not None:
                # 預測評估
                y_pred = beta0 + X_test @ beta
                mae = np.mean(np.abs(y_test - y_pred))
                mae_values.append(mae)
                
                # 收集預測結果
                all_predictions.extend(y_pred.tolist())
                all_actuals.extend(y_test.tolist())
                
                # 係數分析
                nz_coef = np.sum(np.abs(beta) > 0.001)
                coef_vals = {feature: beta[i] for i, feature in enumerate(all_features)}
                fold_results.append({
                    'MAE': mae,
                    'nz_coef': nz_coef,
                    'obj_val': obj_val,
                    'coefs': coef_vals
                })
        
        if mae_values:
            avg_mae = np.mean(mae_values)
            results[B] = avg_mae
            coef_analysis[B] = fold_results
            detailed_results[B]['fold_maes'] = mae_values
            detailed_results[B]['predictions'] = all_predictions
            detailed_results[B]['actuals'] = all_actuals
            print(f"Budget {B}: CV-MAE = {avg_mae:.6f}")
    
    # 選擇最優預算
    best_budget = min(results, key=results.get) if results else None
    return results, best_budget, coef_analysis, detailed_results

# 8. 改進的預測區間建模（GARCH改進, Bonus）
def part_iii_prediction_intervals(data, predictors, knots_dict, best_budget):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    coverages = []
    msis_values = []
    alpha = 0.10
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        # 添加hinge特徵
        train_hinge = add_all_hinge_features(train, knots_dict)
        test_hinge = add_all_hinge_features(test, knots_dict)
        
        # 識別特徵列
        base_features = predictors.copy()
        all_columns = list(train_hinge.columns)
        hinge_features = [col for col in all_columns if col.startswith('H_')]
        all_features = base_features + hinge_features
        
        # 準備資料
        X_train = train_hinge[all_features].values
        y_train = train_hinge['LogReturn'].values
        X_test = test_hinge[all_features].values
        y_test = test_hinge['LogReturn'].values
        
        # 擬合主模型
        beta0, beta, _ = fit_spline_l1_model(X_train, y_train, budget=best_budget)
        
        if beta0 is None:
            continue
            
        # 訓練集預測
        y_pred_train = beta0 + X_train @ beta
        y_pred_test = beta0 + X_test @ beta
        residuals = y_train - y_pred_train
        
        # 改進的變異性建模 (GARCH型)
        residual_sq = residuals**2
        X_var = np.column_stack([
            np.ones(len(y_pred_train)),
            y_pred_train**2,
            np.pad(np.abs(residuals)[:-1], (1, 0), 'constant', constant_values=0)  # 修復shift問題
        ])
        
        try:
            var_model = sm.OLS(residual_sq, X_var).fit()
            X_var_test = np.column_stack([
                np.ones(len(y_pred_test)),
                y_pred_test**2,
                np.abs(residuals[-1]) * np.ones(len(y_pred_test))  # 使用最後一個殘差
            ])
            sigma_sq = var_model.predict(X_var_test)
            sigma_sq = np.maximum(sigma_sq, 1e-6)  # 防止負值
        except:
            # 如果建模失敗，使用簡單方法
            sigma_sq = np.var(residuals) * np.ones(len(y_pred_test))

        # Laplace尺度參數
        b_scale = np.sqrt(sigma_sq) / np.sqrt(2)
        
        # 生成預測區間
        lower_bounds, upper_bounds = [], []
        y_mean_fold = np.mean(y_train)
        
        for i in range(len(y_test)):
            # 生成誤差
            simulated_errors = laplace.rvs(scale=b_scale[i], size=100)
            simulated_y = y_pred_test[i] + simulated_errors
            
            # 計算分位數
            lower_bounds.append(np.percentile(simulated_y, alpha/2 * 100))
            upper_bounds.append(np.percentile(simulated_y, 100 - alpha/2 * 100))
            
            # MSIS計算
            width = upper_bounds[i] - lower_bounds[i]
            penalty_low = max(0, lower_bounds[i] - y_test[i]) * (2 / alpha)
            penalty_high = max(0, y_test[i] - upper_bounds[i]) * (2 / alpha)
            msis = (width + penalty_low + penalty_high) / abs(y_mean_fold) if y_mean_fold != 0 else width
            msis_values.append(msis)
        
        # 覆蓋率計算
        in_interval = (y_test >= np.array(lower_bounds)) & (y_test <= np.array(upper_bounds))
        coverage = np.mean(in_interval)
        coverages.append(coverage)
        print(f"Fold {fold+1}: Coverage = {coverage:.4f}, Mean MSIS = {np.mean(msis_values[-len(y_test):]):.4f}")
    
    return np.mean(coverages) if coverages else np.nan, np.mean(msis_values) if msis_values else np.nan

def create_visualizations(ticker, results_dict, detailed_results, save_dir):
    """創建預測vs實際對比圖和MAE走勢圖"""
    
    # 創建子目錄
    vis_dir = os.path.join(save_dir, f"{ticker}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. MAE走勢圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE vs Budget
    budgets = list(results_dict.keys())
    maes = list(results_dict.values())
    
    ax1.plot(budgets, maes, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Budget Constraint', fontsize=12)
    ax1.set_ylabel('Cross-Validation MAE', fontsize=12)
    ax1.set_title(f'{ticker}: MAE vs Budget Constraint', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 標記最佳預算
    best_budget = min(results_dict, key=results_dict.get)
    best_mae = results_dict[best_budget]
    ax1.axvline(x=best_budget, color='red', linestyle='--', alpha=0.7, label=f'Best Budget: {best_budget}')
    ax1.axhline(y=best_mae, color='red', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Fold-wise MAE distribution for best budget
    if best_budget in detailed_results:
        fold_maes = detailed_results[best_budget]['fold_maes']
        ax2.boxplot([fold_maes], labels=[f'Budget {best_budget}'])
        ax2.scatter([1] * len(fold_maes), fold_maes, alpha=0.6, color='orange')
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title(f'{ticker}: MAE Distribution Across Folds\n(Best Budget: {best_budget})', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{ticker}_mae_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 預測vs實際散點圖
    if best_budget in detailed_results:
        predictions = detailed_results[best_budget]['predictions']
        actuals = detailed_results[best_budget]['actuals']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 散點圖
        ax1.scatter(actuals, predictions, alpha=0.6, s=20, color='steelblue')
        
        # 完美預測線
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # 計算R²
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        r_squared = correlation ** 2
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        
        ax1.set_xlabel('Actual Log Returns', fontsize=12)
        ax1.set_ylabel('Predicted Log Returns', fontsize=12)
        ax1.set_title(f'{ticker}: Predicted vs Actual\nR² = {r_squared:.4f}, MAE = {mae:.6f}', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 殘差圖
        residuals = np.array(actuals) - np.array(predictions)
        ax2.scatter(predictions, residuals, alpha=0.6, s=20, color='orange')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Predicted Log Returns', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title(f'{ticker}: Residual Plot\nMean Residual = {np.mean(residuals):.6f}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{ticker}_prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 時間序列預測圖（取樣顯示）
    if best_budget in detailed_results and len(predictions) > 100:
        # 只顯示前100個預測點，避免圖表過於擁擠
        sample_size = min(100, len(predictions))
        sample_indices = np.linspace(0, len(predictions)-1, sample_size, dtype=int)
        
        sample_actuals = [actuals[i] for i in sample_indices]
        sample_predictions = [predictions[i] for i in sample_indices]
        
        plt.figure(figsize=(14, 8))
        
        plt.plot(range(sample_size), sample_actuals, 'o-', alpha=0.7, label='Actual', linewidth=1.5, markersize=4)
        plt.plot(range(sample_size), sample_predictions, 's-', alpha=0.7, label='Predicted', linewidth=1.5, markersize=4)
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Log Returns', fontsize=12)
        plt.title(f'{ticker}: Time Series Prediction (Sample of {sample_size} points)\nBest Budget: {best_budget}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{ticker}_time_series_prediction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ 可視化圖表已保存至: {vis_dir}")

# 10. 創建總結報告
def create_summary_visualization(final_results, save_dir):
    """創建所有股票的總結對比圖"""
    df = pd.DataFrame(final_results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MAE比較
    x_pos = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x_pos - width/2, df['CV-MAE_Uncon'], width, label='Unconstrained', alpha=0.8, color='lightcoral')
    ax1.bar(x_pos + width/2, df['CV-MAE_Best'], width, label='Best Budget', alpha=0.8, color='steelblue')
    
    ax1.set_xlabel('Stock Ticker', fontsize=12)
    ax1.set_ylabel('Cross-Validation MAE', fontsize=12)
    ax1.set_title('MAE Comparison: Unconstrained vs Best Budget', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['Ticker'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 最佳預算分佈
    budget_counts = df['Best_Budget'].value_counts().sort_index()
    ax2.bar(budget_counts.index.astype(str), budget_counts.values, alpha=0.8, color='lightgreen')
    ax2.set_xlabel('Best Budget Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Best Budget Values', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 覆蓋率比較
    ax3.bar(df['Ticker'], df['Coverage'], alpha=0.8, color='gold')
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.8, label='Target Coverage (90%)')
    ax3.set_xlabel('Stock Ticker', fontsize=12)
    ax3.set_ylabel('Prediction Interval Coverage', fontsize=12)
    ax3.set_title('Prediction Interval Coverage by Stock', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MSIS比較
    ax4.bar(df['Ticker'], df['MSIS'], alpha=0.8, color='mediumpurple')
    ax4.set_xlabel('Stock Ticker', fontsize=12)
    ax4.set_ylabel('Mean Scaled Interval Score (MSIS)', fontsize=12)
    ax4.set_title('MSIS by Stock (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 總結對比圖已保存至: {os.path.join(save_dir, 'summary_comparison.png')}")

def analyze_top_features(coef_analysis, best_budget, all_features, top_k=5):
    """
    分析最重要的特徵
    
    Parameters:
    - coef_analysis: 係數分析結果
    - best_budget: 最佳預算
    - all_features: 所有特徵名稱列表
    - top_k: 回傳前k個重要特徵
    """
    
    if best_budget not in coef_analysis:
        return None
    
    # 收集所有fold的係數
    all_coefs = {}
    for fold_result in coef_analysis[best_budget]:
        for feature, coef_val in fold_result['coefs'].items():
            if feature not in all_coefs:
                all_coefs[feature] = []
            all_coefs[feature].append(coef_val)
    
    # 計算每個特徵的平均係數和重要性
    feature_importance = {}
    for feature, coef_list in all_coefs.items():
        avg_coef = np.mean(coef_list)
        abs_avg_coef = abs(avg_coef)
        std_coef = np.std(coef_list)
        
        feature_importance[feature] = {
            'avg_coef': avg_coef,
            'abs_coef': abs_avg_coef,
            'std_coef': std_coef,
            'importance_score': abs_avg_coef  # 以絕對值作為重要性指標
        }
    
    # 按重要性排序
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1]['importance_score'], 
        reverse=True
    )
    
    # 取前k個重要特徵
    top_features = sorted_features[:top_k]
    
    return top_features, feature_importance

def create_top_features_visualization(ticker, top_features, save_dir):
    """創建Top Features可視化圖表"""
    
    if not top_features:
        return
    
    # 準備數據
    feature_names = [item[0] for item in top_features]
    importance_scores = [item[1]['importance_score'] for item in top_features]
    avg_coefs = [item[1]['avg_coef'] for item in top_features]
    
    # 簡化特徵名稱顯示
    display_names = []
    for name in feature_names:
        if name.startswith('H_'):
            # Hinge特徵簡化顯示
            parts = name.split('_')
            if len(parts) >= 3:
                display_names.append(f"{parts[1]}_H{parts[2][:4]}")
            else:
                display_names.append(name[:15])
        else:
            display_names.append(name)
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 特徵重要性條形圖
    colors = ['red' if coef < 0 else 'green' for coef in avg_coefs]
    bars = ax1.barh(range(len(display_names)), importance_scores, color=colors, alpha=0.7)
    
    ax1.set_yticks(range(len(display_names)))
    ax1.set_yticklabels(display_names)
    ax1.set_xlabel('Feature Importance (|Coefficient|)', fontsize=12)
    ax1.set_title(f'{ticker}: Top {len(top_features)} Most Important Features', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 添加數值標籤
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        ax1.text(score + max(importance_scores) * 0.01, i, f'{score:.4f}', 
                va='center', fontsize=10)
    
    # 2. 係數值（含正負）
    colors_signed = ['red' if coef < 0 else 'green' for coef in avg_coefs]
    bars2 = ax2.barh(range(len(display_names)), avg_coefs, color=colors_signed, alpha=0.7)
    
    ax2.set_yticks(range(len(display_names)))
    ax2.set_yticklabels(display_names)
    ax2.set_xlabel('Average Coefficient Value', fontsize=12)
    ax2.set_title(f'{ticker}: Coefficient Direction & Magnitude', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加數值標籤
    for i, (bar, coef) in enumerate(zip(bars2, avg_coefs)):
        offset = max(abs(min(avg_coefs)), abs(max(avg_coefs))) * 0.02
        x_pos = coef + offset if coef >= 0 else coef - offset
        ax2.text(x_pos, i, f'{coef:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存圖表
    vis_dir = os.path.join(save_dir, f"{ticker}_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, f'{ticker}_top_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {ticker} Top Features 分析圖已保存")

def generate_feature_interpretation(ticker, top_features):
    """生成特徵解釋報告"""
    
    print(f"\n📊 {ticker} - Top Features Analysis:")
    print("="*50)
    
    for i, (feature_name, stats) in enumerate(top_features, 1):
        coef = stats['avg_coef']
        importance = stats['importance_score']
        
        # 特徵類型判斷
        if feature_name.startswith('H_'):
            feature_type = "Spline Hinge Feature"
            base_feature = feature_name.split('_')[1]
            knot_value = feature_name.split('_')[2][:6]
        else:
            feature_type = "Base Feature"
            base_feature = feature_name
            knot_value = ""
        
        # 影響方向
        direction = "正向影響 ↗️" if coef > 0 else "負向影響 ↘️"
        
        print(f"{i}. {feature_name}")
        print(f"   類型: {feature_type}")
        if knot_value:
            print(f"   節點值: {knot_value}")
        print(f"   係數: {coef:.6f}")
        print(f"   重要性: {importance:.6f}")
        print(f"   影響: {direction}")
        
        # 經濟意義解釋
        if base_feature == 'Return_lag1':
            print(f"   意義: 前一日收益率對今日收益的{'正向' if coef > 0 else '反向'}影響")
        elif base_feature == 'Volatility':
            print(f"   意義: 波動率對收益率產生{'正向' if coef > 0 else '負向'}效應")
        elif base_feature == 'RSI':
            print(f"   意義: RSI技術指標呈現{'買入' if coef > 0 else '賣出'}信號")
        elif base_feature == 'LogVolume_lag1':
            print(f"   意義: 前日交易量對今日收益{'正向' if coef > 0 else '負向'}預測")
        elif 'MA' in base_feature:
            print(f"   意義: 移動平均線呈現{'多頭' if coef > 0 else '空頭'}趨勢信號")
        
        print()

# 11. 主程式
def main():
    final_results = []
    predictors = [
        'Return_lag1', 'Return_lag2', 'LogVolume_lag1', 
        'Volatility', 'RSI', 'MA5', 'MA20'
    ]
    
    print("🚀 開始股票收益預測分析...")
    print("="*60)
    
    for ticker in TICKERS:
        print(f"\n======= Analyzing {ticker} =======")
        try:
            data = prepare_data(ticker)
            
            # Part I: 多元預測器無約束回歸
            cv_mae_uncon, knots_dict = part_i_unconstrained(data, predictors)
            print(f"✅ Part I: CV-MAE = {cv_mae_uncon:.6f}")
            
            # Part II: 預算約束優化
            results, best_budget, coef_analysis, detailed_results = part_ii_budgeted(
                data, predictors, knots_dict, BUDGET_LEVELS
            )
            cv_mae_best = results.get(best_budget, np.nan)
            print(f"✅ Part II: Best budget = {best_budget}, CV-MAE = {cv_mae_best:.6f}")
            
            # 先創建 final_results 條目
            result_entry = {
                'Ticker': ticker,
                'CV-MAE_Uncon': cv_mae_uncon,
                'Best_Budget': best_budget,
                'CV-MAE_Best': cv_mae_best,
                'Coverage': np.nan,
                'MSIS': np.nan
            }
            
            # 輸出重要係數解讀
            if best_budget and coef_analysis.get(best_budget):
                coef_sample = coef_analysis[best_budget][0]['coefs']
                print("\nSignificant coefficients:")
                for feature, coef_val in sorted(coef_sample.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    if abs(coef_val) > 0.001:
                        print(f"  {feature}: {coef_val:.4f}")

                # Top Features 分析
                # 重新構建hinge特徵以獲取正確的特徵名稱
                train_sample = add_all_hinge_features(data, knots_dict)
                base_features = predictors.copy()
                all_columns = list(train_sample.columns)
                hinge_features = [col for col in all_columns if col.startswith('H_')]
                all_features = base_features + hinge_features
                
                # 分析Top Features
                top_features, feature_importance = analyze_top_features(
                    coef_analysis, best_budget, all_features, top_k=5
                )
                
                if top_features:
                    # 創建可視化
                    create_top_features_visualization(ticker, top_features, SAVE_DIR)
                    
                    # 生成解釋報告
                    generate_feature_interpretation(ticker, top_features)
                    
                    # 更新結果條目
                    result_entry['Top_Feature'] = top_features[0][0]  # 最重要特徵名稱
                    result_entry['Top_Importance'] = top_features[0][1]['importance_score']  # 重要性分數

            # Part III: 預測區間
            coverage, msis = part_iii_prediction_intervals(
                data, predictors, knots_dict, best_budget
            )
            print(f"✅ Part III: Coverage = {coverage:.4f}, MSIS = {msis:.4f}")

            # 更新Coverage和MSIS
            result_entry['Coverage'] = coverage
            result_entry['MSIS'] = msis

            # 創建可視化
            create_visualizations(ticker, results, detailed_results, SAVE_DIR)
            
            # 保存結果
            final_results.append(result_entry)
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            final_results.append({
                'Ticker': ticker,
                'CV-MAE_Uncon': np.nan,
                'Best_Budget': np.nan,
                'CV-MAE_Best': np.nan,
                'Coverage': np.nan,
                'MSIS': np.nan,
                'Top_Feature': 'N/A',
                'Top_Importance': np.nan
            })

    create_summary_visualization(final_results, SAVE_DIR)
    
    # 保存最終報告
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(os.path.join(SAVE_DIR, "Project_Results.csv"), index=False)
    
    print("\n" + "="*60)
    print("📊 Final Results Summary:")
    print(df_results.to_string(index=False))
    print(f"\n✅ 數值結果已保存至: {os.path.join(SAVE_DIR, 'Project_Results.csv')}")
    print(f"🎨 所有可視化圖表已保存至: {SAVE_DIR} 目錄")
    
    # 簡要分析總結
    print("\n" + "="*60)
    print("📈 Analysis Summary:")
    
    # MAE改善分析
    df_valid = df_results.dropna(subset=['CV-MAE_Best'])
    if len(df_valid) > 0:
        mae_improvement = ((df_valid['CV-MAE_Uncon'] - df_valid['CV-MAE_Best']) / df_valid['CV-MAE_Uncon'] * 100).mean()
        print(f"   Average MAE improvement with budget constraint: {mae_improvement:.2f}%")
        
        best_performing = df_valid.loc[df_valid['CV-MAE_Best'].idxmin(), 'Ticker']
        worst_performing = df_valid.loc[df_valid['CV-MAE_Best'].idxmax(), 'Ticker']
        print(f"   Best performing stock: {best_performing}")
        print(f"   Most challenging stock: {worst_performing}")
        
        avg_coverage = df_valid['Coverage'].mean()
        print(f"   Average prediction interval coverage: {avg_coverage:.1%}")
        
        # 最佳預算統計
        most_common_budget = df_valid['Best_Budget'].mode().iloc[0] if len(df_valid) > 0 else None
        print(f"   Most common optimal budget: {most_common_budget}")
        
        # Top Features 統計 (Bonus)
        top_features_valid = df_valid.dropna(subset=['Top_Feature'])
        if len(top_features_valid) > 0:
            most_important_overall = top_features_valid.loc[top_features_valid['Top_Importance'].idxmax()]
            print(f"   Most important feature overall: {most_important_overall['Top_Feature']} ({most_important_overall['Ticker']})")

if __name__ == "__main__":
    main()
