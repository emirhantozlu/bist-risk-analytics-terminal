import os
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
import numpy as np

def calculate_turkish_sentiment(text):
    text_lower = text.lower()
    
    pos_dict = {
        'rekor': 1.0, 'kazanç': 0.8, 'büyüme': 0.9, 'artış': 0.7, 'anlaşma': 0.8,
        'olumlu': 0.6, 'temettü': 0.9, 'kar': 0.8, 'kâr': 0.8, 'yükseldi': 0.8,
        'yükseliş': 0.8, 'fırladı': 0.9, 'ortaklık': 0.7, 'güçlü': 0.6, 'zirve': 0.9,
        'prim': 0.6, 'kazandı': 0.7, 'destek': 0.5, 'alım': 0.6, 'ihale': 0.8
    }
    
    neg_dict = {
        'düşüş': -0.7, 'zarar': -0.9, 'kayıp': -0.8, 'ceza': -0.9, 'düştü': -0.8,
        'risk': -0.6, 'olumsuz': -0.7, 'azalma': -0.6, 'geriledi': -0.7, 'çakıldı': -0.9,
        'dava': -0.6, 'haciz': -0.9, 'kriz': -0.9, 'sert': -0.4, 'baskı': -0.5,
        'satış': -0.5, 'taban': -0.8, 'gerileme': -0.7, 'kaybetti': -0.7, 'borç': -0.6
    }
    
    negators = ['değil', 'yok', 'olamadı', 'açıklayamadı', 'başaramadı', 'edemedi', 'düşemedi']
    
    score = 0.0
    words = text_lower.split()
    matched_count = 0
    
    for i, word in enumerate(words):
        clean_word = "".join([c for c in word if c.isalnum()])
        if clean_word in pos_dict:
            current_score = pos_dict[clean_word]
            lookahead = words[i:i+3]
            if any(neg in lookahead for neg in negators):
                current_score = -0.8
            score += current_score
            matched_count += 1
        elif clean_word in neg_dict:
            current_score = neg_dict[clean_word]
            score += current_score
            matched_count += 1
            
    if matched_count > 0:
        final_score = score / matched_count
        return round(max(-1.0, min(1.0, final_score)), 2)
    
    return 0.0

def get_news_and_sentiment(stock_ticker):
    clean_ticker = stock_ticker.split('.')[0]
    url = f"https://news.google.com/rss/search?q={clean_ticker}+hisse+borsa&hl=tr&gl=TR&ceid=TR:tr"
    news_items = []
    
    try:
        req = Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'tr-TR,tr;q=0.9'
        })
        
        with urlopen(req, timeout=10) as response:
            xml_data = response.read()
            tree = ET.ElementTree(ET.fromstring(xml_data))
            root = tree.getroot()
            
            items = root.findall('.//item')
            if len(items) == 0:
                return [{'title': f"{clean_ticker} için anlık haber akışı bulunamadı.", 'date': 'Bilgi', 'score': 0.0, 'link': '#'}]
                
            for item in items[:4]:
                title = item.find('title').text
                if " - " in title:
                    title = " - ".join(title.split(" - ")[:-1])
                    
                pub_date = item.find('pubDate').text[:16] if item.find('pubDate') is not None else 'Güncel'
                
                # Orijinal haber linkini de çekiyoruz
                link = item.find('link').text if item.find('link') is not None else '#'
                
                score = calculate_turkish_sentiment(title)
                
                # Sözlüğe 'link' anahtarını ekledik
                news_items.append({'title': title, 'date': pub_date, 'score': score, 'link': link})
                
    except Exception as e:
        news_items = [{'title': f"Haber çekilemedi (Google RSS): {str(e)}", 'date': 'Hata', 'score': 0.0, 'link': '#'}]
    
    return news_items