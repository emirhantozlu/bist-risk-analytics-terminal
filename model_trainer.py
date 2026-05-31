import os
import joblib
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from data_processor import load_and_process_data

def run_offline_training(stock_code):
    print(f"⚡ {stock_code} için kurumsal eğitim hattı başlatıldı...")
    
    # 1. Güncel veriyi indir ve işle
    df = load_and_process_data(stock_code)
    if df is None or len(df) < 100:
        print(f"❌ {stock_code} için yeterli veri sağlanamadı.")
        return False
        
    # Öznitelikler ve Hedef Değişkenler
    features = ['RSI', 'BB_Width', 'MACD_Hist', 'Vol_Ratio']
    X = df[features]
    y_reg = df['Target_Volatility']
    y_clf = df['Target_Direction']
    
    # Veriyi Eğitim ve Test olarak bölüyoruz
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, shuffle=False)
    
    # Klasör kontrolü
    os.makedirs('saved_models', exist_ok=True)
    
    # 2. Volatilite Regresyon Modellerinin Eğitimi ve Kaydedilmesi
    reg_models = {
        'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    for name, model in reg_models.items():
        model.fit(X_train, y_train_reg)
        # Modeli diske yazıyoruz
        joblib.dump(model, f'saved_models/{stock_code}_{name}_reg.joblib')
        
    # 3. Yön Sınıflandırma Modellerinin Eğitimi ve Kaydedilmesi
    clf_models = {
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0),
        'CatBoost': CatBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=0)
    }
    
    for name, model in clf_models.items():
        model.fit(X_train, y_train_clf)
        # Modeli diske yazıyoruz
        joblib.dump(model, f'saved_models/{stock_code}_{name}_clf.joblib')
        
    print(f"✅ {stock_code} için tüm modeller başarıyla 'saved_models/' klasörüne kaydedildi.")
    return True

if __name__ == "__main__":
    # Test amaçlı manuel çalıştırma hattı
    run_offline_training("THYAO.IS")