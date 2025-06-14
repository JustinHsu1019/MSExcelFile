import yfinance as yf
import numpy as np
import pandas as pd
import os
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
    
    for B in budget_levels:
        mae_values = []
        fold_results = []
        
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
            print(f"Budget {B}: CV-MAE = {avg_mae:.6f}")
    
    # 選擇最優預算
    best_budget = min(results, key=results.get) if results else None
    return results, best_budget, coef_analysis

# 8. 改進的預測區間建模（GARCH改進）
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

# 9. 主程式
def main():
    final_results = []
    predictors = [
        'Return_lag1', 'Return_lag2', 'LogVolume_lag1', 
        'Volatility', 'RSI', 'MA5', 'MA20'
    ]
    
    for ticker in TICKERS:
        print(f"\n======= Analyzing {ticker} =======")
        try:
            data = prepare_data(ticker)
            
            # Part I: 多元預測器無約束回歸
            cv_mae_uncon, knots_dict = part_i_unconstrained(data, predictors)
            print(f"✅ Part I: CV-MAE = {cv_mae_uncon:.6f}")
            
            # Part II: 預算約束優化
            results, best_budget, coef_analysis = part_ii_budgeted(
                data, predictors, knots_dict, BUDGET_LEVELS
            )
            cv_mae_best = results.get(best_budget, np.nan)
            print(f"✅ Part II: Best budget = {best_budget}, CV-MAE = {cv_mae_best:.6f}")
            
            # 輸出重要係數解讀
            if best_budget and coef_analysis.get(best_budget):
                coef_sample = coef_analysis[best_budget][0]['coefs']
                print("\nSignificant coefficients:")
                for feature, coef_val in sorted(coef_sample.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    if abs(coef_val) > 0.001:
                        print(f"  {feature}: {coef_val:.4f}")
            
            # Part III: 預測區間
            coverage, msis = part_iii_prediction_intervals(
                data, predictors, knots_dict, best_budget
            )
            print(f"✅ Part III: Coverage = {coverage:.4f}, MSIS = {msis:.4f}")
            
            # 保存結果
            final_results.append({
                'Ticker': ticker,
                'CV-MAE_Uncon': cv_mae_uncon,
                'Best_Budget': best_budget,
                'CV-MAE_Best': cv_mae_best,
                'Coverage': coverage,
                'MSIS': msis
            })
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            final_results.append({
                'Ticker': ticker,
                'CV-MAE_Uncon': np.nan,
                'Best_Budget': np.nan,
                'CV-MAE_Best': np.nan,
                'Coverage': np.nan,
                'MSIS': np.nan
            })
    
    # 保存最終報告
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(os.path.join(SAVE_DIR, "Project_Results.csv"), index=False)
    print("\n" + "="*50)
    print("📊 Final Results Summary:")
    print(df_results.to_string(index=False))
    print(f"\n✅ Results saved to {os.path.join(SAVE_DIR, 'Project_Results.csv')}")

if __name__ == "__main__":
    main()
