# ==============================================================================
#             BIST RISK & TREND ANALYTICS TERMINAL - CONFIGURATION
# ==============================================================================

# Veri Hattı Ayarları
DATA_SETTINGS = {
    "PERIOD": "5y",                  # yfinance üzerinden çekilecek tarihsel derinlik
    "EWMA_LAMBDA": 0.94,             # RiskMetrics standart oynaklık ceza faktörü
    "DEFAULT_STOCKS": ["THYAO", "EREGL", "ASELS"], # Çoklu mod başlangıç listesi
}

# Makine Öğrenmesi Model Hiper-Parametreleri
MODEL_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "random_state": 42,
}

# Teknik İndikatör Ayarları
INDICATOR_WINDOWS = {
    "RSI": 14,
    "BOLLINGER": 20,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "VOLUME_SMA": 5
}