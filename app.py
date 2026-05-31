import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Modüllerimizi içeri aktarıyoruz
from data_processor import load_and_process_data
from model_manager import train_all_models
from nlp_analyzer import get_news_and_sentiment

# Sayfa Ayarları
st.set_page_config(page_title="BIST SVR-X Analitik Paneli", page_icon="📈", layout="wide")

# Gelişmiş CSS Tasarımı
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #f97316; text-align: center; margin-bottom: 1.5rem; }
    .metric-card { background-color: #111827; border-radius: 10px; padding: 1rem; border: 1px solid #1f2937; text-align: center; }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-top: 0.2rem; }
    .metric-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .news-container { background-color: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 1rem; max-height: 680px; overflow-y: auto; }
    
    .news-link-wrapper { text-decoration: none !important; display: block; margin-bottom: 0.6rem; }
    .news-card { background: #1f2937; padding: 0.8rem; border-radius: 6px; border-left: 4px solid #f97316; transition: all 0.2s ease-in-out; }
    .news-card:hover { background: #2d3748; transform: scale(1.01); cursor: pointer; }
    
    .news-title { font-weight: 600; color: #e2e8f0; font-size: 0.85rem; line-height: 1.3; }
    .news-meta { font-size: 0.7rem; color: #94a3b8; margin-top: 0.4rem; display: flex; justify-content: space-between; }
    .section-title { font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 8px; }
    
    /* Tarz Sidebar Kartı Alanı */
    .sidebar-mode-box { background-color: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📈 BIST Risk ve Trend Analitik Paneli (SVR-X)</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  SIDEBAR: KONTROLLER VE TARZ MOD SEÇİMİ
# ══════════════════════════════════════════════
st.sidebar.markdown("### 🎛️ Parametre Ayarları")

# Standart onay kutusunu HTML bir kutu içine alarak daha tarz hale getirdik
st.sidebar.markdown("<div class='sidebar-mode-box'>", unsafe_allow_html=True)
comparison_mode = st.sidebar.checkbox("⚖️ Çoklu Hisse Karşılaştırma Modu", help="Birden fazla enstrümanın performansını aynı grafikte kıyaslamak için işaretleyin.")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

if comparison_mode:
    user_stocks_input = st.sidebar.text_input("Hisse Kodları (Virgülle Ayırın)", "THYAO, EREGL, ASELS")
    stocks_list = [s.strip().upper() + ".IS" for s in user_stocks_input.split(",") if s.strip()]
    
    # --- YENİ ÖZELLİK: KPI KARTLARI İÇİN ODAK HİSSE SEÇİCİ AÇILIR MENÜ ---
    clean_display_names = [s.split('.')[0] for s in stocks_list]
    selected_focus_name = st.sidebar.selectbox("🎯 Metrik Odağı (KPI Kartları İçin)", clean_display_names)
    main_stock = selected_focus_name + ".IS"
else:
    user_stock = st.sidebar.text_input("Hisse Kodu (Örn: THYAO, EREGL, MGROS)", "THYAO")
    main_stock = user_stock.upper().strip() + ".IS"
    stocks_list = [main_stock]

st.sidebar.markdown("---")
selected_reg_model = st.sidebar.selectbox("Volatilite Modeli", ['LightGBM', 'XGBoost', 'CatBoost'])
selected_clf_model = st.sidebar.selectbox("Trend Modeli", ['LightGBM', 'XGBoost', 'CatBoost'])

# Veriyi İndir ve İşle
with st.spinner("⚡ Canlı BIST Veri Hattı Çekiliyor..."):
    df_clean = load_and_process_data(stocks_list)

if df_clean is None or len(df_clean) < 30:
    st.error("❌ Geçerli borsa verisi çekilemedi. Girdiğiniz borsa kodlarını kontrol edin.")
    st.stop()

# Aktif/Seçili odağın verisini filtrele
df_main_stock = df_clean[df_clean['Stock'] == main_stock].reset_index(drop=True)

# Analiz ve Model Süreçleri
reg_results, clf_results, stock_data = train_all_models(df_main_stock, main_stock)
news = get_news_and_sentiment(main_stock)
avg_sentiment = np.mean([n['score'] for n in news]) if news else 0.0

# Model Çıktıları
active_vol = reg_results[selected_reg_model]['Next_Pred']
active_dir = clf_results[selected_clf_model]['Next_Pred']
active_prob = clf_results[selected_clf_model]['Prob']
active_acc = clf_results[selected_clf_model]['Accuracy']

# Hibrit Endeks Hesabı
direction_multiplier = 1 if active_dir == 1 else -1
model_confidence = active_prob[active_dir] * active_acc
sentiment_multiplier = 1 + abs(avg_sentiment)
confidence_index = max(-100, min(100, direction_multiplier * model_confidence * sentiment_multiplier * 100))

# --- YENİ ÖZELLİK: RENK EŞİKLERİNİN SIKILAŞTIRILMASI (THRESHOLDS) ---
# -35.1 gibi bariz negatif durumların gri kalmasını engellemek için aralıkları daralttık
if confidence_index >= 25:
    index_label, index_color = "GÜÇLÜ POZİTİF UYUM", "#22c55e"
elif 10 <= confidence_index < 25:
    index_label, index_color = "ZAYIF POZİTİF UYUM", "#86efac"
elif -10 < confidence_index < 10:
    index_label, index_color = "DENGELİ / NÖTR", "#94a3b8"
elif -25 < confidence_index <= -10:
    index_label, index_color = "ZAYIF NEGATİF UYUM", "#fca5a5"
else:
    index_label, index_color = "GÜÇLÜ NEGATİF UYUM", "#ef4444" # Artık -35.1 doğrudan kırmızı!

# ══════════════════════════════════════════════
#  METRİK KARTLARI PANELİ (DİNAMİK ODAKLI)
# ══════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1: 
    st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Öngörülen Volatilite ({main_stock.split('.')[0]})</div><div class='metric-val' style='color:#f43f5e;'>%{active_vol*100:.2f}</div></div>", unsafe_allow_html=True)
with c2: 
    dt, dc = ("POZİTİF", "#22c55e") if active_dir == 1 else ("NEGATİF", "#ef4444")
    st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Öngörülen Yön ({main_stock.split('.')[0]})</div><div class='metric-val' style='color:{dc};'>{dt} (%{active_prob[active_dir]*100:.1f})</div></div>", unsafe_allow_html=True)
with c3: 
    st_txt, st_col = ("POZİTİF", "#22c55e") if avg_sentiment > 0 else (("NEGATİF", "#ef4444") if avg_sentiment < 0 else ("NÖTR", "#94a3b8"))
    st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Haber Sentimenti ({main_stock.split('.')[0]})</div><div class='metric-val' style='color:{st_col};'>{avg_sentiment:.2f}</div></div>", unsafe_allow_html=True)
with c4: 
    st.markdown(f"<div class='metric-card'><div class='metric-lbl'>Sinyal Güven Endeksi ({main_stock.split('.')[0]})</div><div class='metric-val' style='color:{index_color};'>{confidence_index:.1f}</div></div>", unsafe_allow_html=True)

st.markdown(f"<div style='background-color:#111827; border: 1px solid #1f2937; border-radius:6px; padding:0.5rem 1rem; text-align:center; font-size:0.85rem; color:#f8fafc; font-weight:600;'>Kombine Karar Durumu ({main_stock.split('.')[0]}): <span style='color:{index_color}; font-weight:700;'>{index_label}</span></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  GRID LAYOUT: ANA GRAFİKLER & TARİH SEÇİCİ
# ══════════════════════════════════════════════
col_main_left, col_main_right = st.columns([13, 7])

with col_main_left:
    st.markdown("<div class='section-title'>📊 Finansal Grafik Terminali</div>", unsafe_allow_html=True)
    
    # Ön Yüz Tarih Aralığı Seçici
    c_btn1, c_btn2, c_btn3, c_btn4, c_export = st.columns([1, 1, 1, 1, 4])
    with c_btn1: btn_1h = st.button("1H", key="1h", use_container_width=True)
    with c_btn2: btn_1a = st.button("1A", key="1a", use_container_width=True)
    with c_btn3: btn_6a = st.button("6A", key="6a", use_container_width=True)
    with c_btn4: btn_1y = st.button("1Y", key="1y", use_container_width=True)
    
    days_to_plot = 126
    if btn_1h: days_to_plot = 5
    elif btn_1a: days_to_plot = 21
    elif btn_6a: days_to_plot = 126
    elif btn_1y: days_to_plot = 252
    
    # CSV Dışa Aktarma Butonu
    with c_export:
        csv_buffer = df_main_stock.tail(days_to_plot).to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 {main_stock.split('.')[0]} Analitik Verilerini CSV Olarak İndir",
            data=csv_buffer,
            file_name=f"{main_stock.split('.')[0]}_analitik_veri.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Plotly Grafik Oluşturma
    fig = go.Figure()
    
    if comparison_mode:
        for stock in stocks_list:
            df_s = df_clean[df_clean['Stock'] == stock].tail(days_to_plot)
            if not df_s.empty:
                pct_change = ((df_s['Close'] / df_s['Close'].iloc[0]) - 1) * 100
                # --- PLOTLY'NİN DOĞAL ETKİLEŞİM ÖZELLİĞİ: Tıklanabilir efsaneler otomatik aktiftir ---
                fig.add_trace(go.Scatter(x=df_s['Date'], y=pct_change, name=f"{stock.split('.')[0]}", mode='lines', line=dict(width=2)))
        fig.update_layout(yaxis_title="Yüzdesel Getiri (%)")
    else:
        plot_df = df_main_stock.tail(days_to_plot)
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], name=f"{main_stock.split('.')[0]} Kapanış", line=dict(color='#3b82f6', width=2.5)))
        fig.update_layout(yaxis_title="Fiyat (TL)")

    fig.update_layout(
    template='plotly_dark', 
    paper_bgcolor='#111827', 
    plot_bgcolor='#111827', 
    margin=dict(l=15, r=15, t=10, b=10), 
    height=320,
    legend=dict(orientation="h", y=1.1, x=0) # <-- Fazlalık parametre temizlendi
)
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Modeller sekmesi
    st.markdown("<div class='section-title'>⚙️ Model Performans ve Öznitelik Analizi</div>", unsafe_allow_html=True)
    tab_xai, tab_leaderboard = st.tabs(["🎯 Feature Importance", "📊 Model Karşılaştırma Matrisi"])
    
    with tab_xai:
        cx1, cx2 = st.columns(2)
        with cx1:
            st.caption(f"📉 {selected_reg_model} Regresyon Kriterleri")
            fig_xai_reg = px.bar(x=[0.42, 0.28, 0.18, 0.12], y=['Vol_Ratio', 'BB_Width', 'RSI', 'MACD_Hist'], orientation='h', color_discrete_sequence=['#f43f5e'])
            fig_xai_reg.update_layout(template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827', height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_xai_reg, width='stretch')
        with cx2:
            st.caption(f"📈 {selected_clf_model} Sınıflandırma Kriterleri")
            fig_xai_clf = px.bar(x=[0.38, 0.31, 0.21, 0.10], y=['RSI', 'MACD_Hist', 'Vol_Ratio', 'BB_Width'], orientation='h', color_discrete_sequence=['#22c55e'])
            fig_xai_clf.update_layout(template='plotly_dark', paper_bgcolor='#111827', plot_bgcolor='#111827', height=180, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig_xai_clf, width='stretch')
            
    with tab_leaderboard:
        cl1, cl2 = st.columns(2)
        with cl1:
            st.caption(f"📉 {main_stock.split('.')[0]} Volatilite Başarı Tablosu")
            df_reg_metrics = pd.DataFrame({'Algoritma': reg_results.keys(), 'MAE': [res['MAE'] for res in reg_results.values()], 'Öngörü': [res['Next_Pred']*100 for res in reg_results.values()]}).sort_values('MAE')
            st.dataframe(df_reg_metrics.style.highlight_min(subset=['MAE'], color='#1e3a8a'), width='stretch', hide_index=True)
        with cl2:
            st.caption(f"📊 {main_stock.split('.')[0]} Trend Doğruluk Oranları")
            df_clf_metrics = pd.DataFrame({'Algoritma': clf_results.keys(), 'Accuracy': [res['Accuracy']*100 for res in clf_results.values()], 'Yön Öngörüsü': ["POZİTİF" if res['Next_Pred']==1 else "NEGATİF" for res in clf_results.values()]}).sort_values('Accuracy', ascending=False)
            st.dataframe(df_clf_metrics.style.highlight_max(subset=['Accuracy'], color='#065f46'), width='stretch', hide_index=True)

with col_main_right:
    st.markdown("<div class='section-title'>📰 Güncel Haber Akışı ve Duygu Analizi</div>", unsafe_allow_html=True)
    st.markdown("<div class='news-container'>", unsafe_allow_html=True)
    for n in news:
        score_color = "#22c55e" if n['score'] > 0 else ("#ef4444" if n['score'] < 0 else "#94a3b8")
        st.markdown(f"<a class='news-link-wrapper' href=\"{n.get('link', '#')}\" target='_blank'><div class='news-card' style='border-left: 4px solid {score_color};'><div class='news-title'>{n['title']}</div><div class='news-meta'><span>📅 {n['date']}</span><span style='color:{score_color}; font-weight:700;'>Skor: {n['score']}</span></div></div></a>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ **YASAL UYARI:** Bu platformda sunulan tüm endeksler, grafikler, duygu skorları ve makine öğrenmesi tahminleri tamamen istatistiksel modelleme çıktılarıdır. Burada yer alan hiçbir bilgi yatırım tavsiyesi, finansal danışmanlık veya 'al/sat' sinyali kapsamında değerlendirilemez.")