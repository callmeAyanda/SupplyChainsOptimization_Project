# event_detector.py
from transformers import pipeline
import re
import pandas as pd
import numpy as np

class ZAEventDetector:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased"
        )
        
        self.loc_patterns = {
            'PORT': r"(Durban|Cape Town|Port Elizabeth|Richards Bay|Saldanha Bay|East London)",
            'MINE': r"(Rustenburg|Bushveld|Witwatersrand|Sishen|Kathu|Carletonville|Welkom)",
            'HIGHWAY': r"(N1|N2|N3|N4|N5|N6|N7|R21|R300|R101|R102)",
            'BORDER': r"(Beitbridge|Lebombo|Kopfontein|Maseru Bridge|Oshoek)"
        }
        
        self.event_types = {
            'strike': r"(strike|walkout|protest|industrial action)",
            'flood': r"(flood|rain|storm|deluge|inundation)",
            'disruption': r"(disruption|delay|closure|blockade|obstruction)",
            'crime': r"(hijack|theft|robbery|vandalism|sabotage)"
        }
    
    def extract_sa_entities(self, text):
        """Identify SA-specific locations and event types"""
        entities = {'location': None, 'event_type': None, 'severity': 0}
        
        # Detect locations
        for entity_type, pattern in self.loc_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['location'] = match.group(0)
                break
        
        # Detect event types
        for event, pattern in self.event_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities['event_type'] = event
                break
        
        # Classify event severity
        result = self.classifier(text[:512])
        if result[0]['label'] == 'POSITIVE':
            entities['severity'] = max(0, min(1, result[0]['score'] * 0.3))  # Positive news less severe
        else:
            entities['severity'] = min(1, result[0]['score'] * 1.2)  # Negative news more severe
        
        return entities
    
    def generate_risk_score(self, events_df):
        """Calculate disruption scores for SA routes"""
        if events_df.empty:
            return pd.DataFrame(columns=['location', 'risk_score', 'latest_event', 'event_count', 'lat', 'lon'])
        
        # Ensure we have the required columns
        if 'weighted_severity' not in events_df.columns:
            events_df['weighted_severity'] = events_df.get('severity', 0) * events_df.get('time_decay', 1)

        # Weight recent events higher
        events_df['time_decay'] = np.exp(-0.2 * (pd.Timestamp.now() - events_df['timestamp']).dt.days)
        events_df['weighted_severity'] = events_df['severity'] * events_df['time_decay']
        
        # Apply event type multipliers
        event_weights = {
            'strike': 1.3,
            'flood': 1.5,
            'disruption': 1.2,
            'crime': 1.4
        }
        events_df['weighted_severity'] *= events_df['event_type'].map()
        lambda x: event_weights.get(x, 1.0)
        
        
        # Aggregate by location
        risk_scores = events_df.groupby('location').agg(
            risk_score=('weighted_severity', 'sum'),
            latest_event=('timestamp', 'max'),
            event_count=('severity', 'count')
        ).reset_index()
        
        # Normalize scores 0-1
        if not risk_scores.empty:
            max_score = risk_scores['risk_score'].max()
            if max_score > 0:
                risk_scores['risk_score'] = risk_scores['risk_score'] / max_score
        
        return risk_scores.sort_values('risk_score', ascending=False)
    
    def process_news(self, news_df):
        """Process news dataframe and extract risk info"""
        if news_df.empty:
            return pd.DataFrame()
        
        events = []
        for _, row in news_df.iterrows():
            text = f"{row['title']}. {row['summary']}"
            entities = self.extract_sa_entities(text)
            if entities['location']:
                events.append({
                    'timestamp': row['timestamp'],
                    'location': entities['location'],
                    'event_type': entities['event_type'],
                    'severity': entities['severity'],
                    'source': row['source'],
                    'title': row['title'],
                    'link': row['link']
                })
        return pd.DataFrame(events)
