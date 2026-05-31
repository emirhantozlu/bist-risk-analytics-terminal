import os
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score
from model_trainer import run_offline_training

def train_all_models(df, stock_code):
    features = ['RSI', 'BB_Width', 'MACD_Hist', 'Vol_Ratio']
    
    # Eğitim ve test setlerini arayüz metrik hesabı için tekrar hizalayalım
    X = df[features]
    y_reg = df['Target_Volatility']
    y_clf = df['Target_Direction']
    
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test_reg = y_reg.iloc[split_idx:]
    y_test_clf = y_clf.iloc[split_idx:]
    
    # En son güncel veri satırı (Yarınki tahmini üreteceğimiz girdi)
    latest_features = X.iloc[[-1]]
    
    model_names = ['LightGBM', 'XGBoost', 'CatBoost']
    reg_results = {}
    clf_results = {}
    
    # Disk kontrolü: Eğer modeller yoksa otomatik eğit
    if not os.path.exists(f'saved_models/{stock_code}_LightGBM_reg.joblib'):
        run_offline_training(stock_code)
        
    # --- VOLATİLİTE MODELLERİNİ DİSKTEN OKUMA (INFERENCE) ---
    for name in model_names:
        model = joblib.load(f'saved_models/{stock_code}_{name}_reg.joblib')
        preds = model.predict(X_test)
        next_pred = model.predict(latest_features)[0]
        mae = mean_absolute_error(y_test_reg, preds)
        
        reg_results[name] = {
            'MAE': round(mae, 4),
            'Next_Pred': next_pred
        }
        
    # --- YÖN MODELLERİNİ DİSKTEN OKUMA (INFERENCE) ---
    for name in model_names:
        model = joblib.load(f'saved_models/{stock_code}_{name}_clf.joblib')
        preds = model.predict(X_test)
        next_pred = model.predict(latest_features)[0]
        prob = model.predict_proba(latest_features)[0]
        acc = accuracy_score(y_test_clf, preds)
        
        clf_results[name] = {
            'Accuracy': round(acc, 4),
            'Next_Pred': int(next_pred),
            'Prob': prob
        }
        
    return reg_results, clf_results, df