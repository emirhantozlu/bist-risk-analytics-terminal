import os
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_data(stock_codes):
    """
    Tek veya birden fazla hisse kodunu yfinance üzerinden list olarak alır,
    canlı indirir ve kurumsal EWMA finansal matematik özniteliklerini hesaplar.
    """
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]
        
    all_data = []
    
    for code in stock_codes:
        try:
            ticker = yf.Ticker(code)
            group = ticker.history(period="5y")
            
            if group.empty:
                continue
                
            group = group.reset_index()
            group['Stock'] = code
            group = group[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock']]
            group['Date'] = pd.to_datetime(group['Date']).dt.tz_localize(None)
            
            group = group.sort_values('Date').reset_index(drop=True)
            
            # --- FİNANSAL MATEMATİK VE ÖZNİTELİK MÜHENDİSLİĞİ ---
            group['Log_Returns'] = np.log(group['Close'] / group['Close'].shift(1))
            
            # EWMA Volatilite (Lambda = 0.94)
            lambda_val = 0.94
            sq_returns = group['Log_Returns'] ** 2
            ewma_var = sq_returns.ewm(alpha=(1 - lambda_val), adjust=False).mean()
            group['Target_Volatility'] = np.sqrt(ewma_var) * np.sqrt(252)
            
            # Trend Yönü
            group['Target_Direction'] = np.where(group['Close'].shift(-1) > group['Close'], 1, 0)
            
            # Teknik Göstergeler
            delta = group['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            group['RSI'] = 100 - (100 / (1 + rs))
            
            rmean = group['Close'].rolling(window=20).mean()
            rstd = group['Close'].rolling(window=20).std()
            group['BB_Width'] = ((rmean + (rstd * 2)) - (rmean - (rstd * 2))) / rmean
            
            exp1 = group['Close'].ewm(span=12, adjust=False).mean()
            exp2 = group['Close'].ewm(span=26, adjust=False).mean()
            group['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()
            
            group['Vol_Ratio'] = group['Volume'] / group['Volume'].rolling(window=5).mean()
            group['EWMA_Lag'] = group['Target_Volatility'].shift(1)
            
            group = group.dropna().reset_index(drop=True)
            all_data.append(group)
            
        except Exception:
            continue
            
    if not all_data:
        return None
        
    return pd.concat(all_data).reset_index(drop=True)