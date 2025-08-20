# data_processor.py
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import requests
from bs4 import BeautifulSoup
import re
import pytz
from datetime import datetime

class SADataProcessor:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="supply_chain_za")
        self.provinces = ['Gauteng', 'Western Cape', 'Eastern Cape', 'KwaZulu-Natal', 
                         'Free State', 'North West', 'Northern Cape', 'Mpumalanga', 'Limpopo']
        self.sa_timezone = pytz.timezone('Africa/Johannesburg')
    
    def _get_current_date(self):
        return datetime.now(self.sa_timezone).date()
    
    def load_freight_data(self):
        """Load real-time freight data from Transnet APIs"""
        try:
            # Transnet API endpoint (sample structure)
            url = "https://api.transnet.co.za/freight/v1/delays"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['records'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['delay_hours'] = df['delay_hours'].astype(float)
            df['route'] = df['origin'] + '-' + df['destination']
            return df
        except:
            # Fallback to sample data if API fails
            print("Using sample freight data")
            return pd.DataFrame({
                'timestamp': pd.date_range(end=self._get_current_date(), periods=30, freq='D'),
                'origin': ['Johannesburg', 'Durban', 'Cape Town', 'Port Elizabeth'] * 7 + ['Johannesburg', 'Durban'],
                'destination': ['Cape Town', 'Johannesburg', 'Durban', 'Johannesburg'] * 7 + ['Durban', 'Cape Town'],
                'delay_hours': np.random.lognormal(1.5, 0.8, 30),
                'cargo_type': ['Mining', 'Agricultural', 'Manufacturing'] * 10
            })
    
    def get_commodity_prices(self):
        """Get SA commodity prices from JSE and global markets"""
        try:
            # African Markets API
            url = "https://africanmarkets.io/api/v1/south-africa/commodities"
            response = requests.get(url)
            data = response.json()
            
            commodities = []
            for item in data['data']:
                commodities.append({
                    'commodity': item['name'],
                    'price': float(item['last_price'].replace(',', '')),
                    'change_pct': float(item['change'].strip('%')),
                    'unit': item['unit']
                })
            return pd.DataFrame(commodities)
        except:
            print("Using sample commodity data")
            return pd.DataFrame({
                'commodity': ['Coal', 'Platinum', 'Gold', 'Maize', 'Wheat'],
                'price': [125.7, 978.5, 1982.3, 350.2, 280.9],
                'change_pct': [1.2, -0.5, 0.8, -2.1, 1.5],
                'unit': ['USD/ton', 'USD/oz', 'USD/oz', 'ZAR/ton', 'ZAR/ton']
            })
    
    def get_sa_news(self):
        """Scrape SA news sites for supply chain relevant events"""
        try:
            articles = []
            # News24 scraping
            url = "https://www.news24.com/southafrica"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.select('article'):
                title = article.select_one('h3 a').text.strip()
                summary = article.select_one('p').text.strip() if article.select_one('p') else ""
                link = article.select_one('h3 a')['href']
                if not link.startswith('http'):
                    link = f"https://www.news24.com{link}"
                
                # Filter for supply chain relevant news
                keywords = ['strike', 'flood', 'protest', 'port', 'load shedding', 
                            'border', 'customs', 'freight', 'rail', 'shipping']
                if any(kw in title.lower() for kw in keywords):
                    articles.append({
                        'title': title,
                        'summary': summary,
                        'link': link,
                        'source': 'News24',
                        'timestamp': datetime.now(self.sa_timezone)
                    })
            return pd.DataFrame(articles)
        except Exception as e:
            print(f"News scraping failed: {e}")
            return pd.DataFrame({
                'title': ['KZN Floods Disrupt Port Operations', 'Transnet Strike Enters Second Week'],
                'summary': ['Heavy rains cause delays at Durban port', 'Freight rail workers demand higher wages'],
                'link': ['#', '#'],
                'source': ['Sample', 'Sample'],
                'timestamp': [datetime.now(self.sa_timezone)] * 2
            })
    
    def geocode_sa_locations(self, location_names):
        """Convert SA location names to coordinates with caching"""
        locations = {}
        cache = {
            'Johannesburg': (-26.2041, 28.0473),
            'Durban': (-29.8587, 31.0218),
            'Cape Town': (-33.9249, 18.4241),
            'Port Elizabeth': (-33.9608, 25.6022),
            'Richards Bay': (-28.7807, 32.0383),
            'East London': (-33.0292, 27.8546)
        }
        
        for loc in location_names:
            if loc in cache:
                locations[loc] = cache[loc]
            else:
                try:
                    result = self.geolocator.geocode(loc + ", South Africa")
                    if result:
                        locations[loc] = (result.latitude, result.longitude)
                    else:
                        locations[loc] = (None, None)
                except:
                    locations[loc] = (None, None)
        return locations
    
    def create_spatial_features(self, df):
        """Add province and economic region features"""
        province_map = {
            'Johannesburg': 'Gauteng',
            'Pretoria': 'Gauteng',
            'Durban': 'KwaZulu-Natal',
            'Pietermaritzburg': 'KwaZulu-Natal',
            'Cape Town': 'Western Cape',
            'Stellenbosch': 'Western Cape',
            'Port Elizabeth': 'Eastern Cape',
            'East London': 'Eastern Cape',
            'Bloemfontein': 'Free State',
            'Kimberley': 'Northern Cape',
            'Rustenburg': 'North West',
            'Nelspruit': 'Mpumalanga',
            'Polokwane': 'Limpopo'
        }
        
        economic_zone_map = {
            'Gauteng': 'Industrial',
            'KwaZulu-Natal': 'Industrial',
            'Western Cape': 'Agricultural',
            'Eastern Cape': 'Agricultural',
            'Free State': 'Agricultural',
            'North West': 'Mineral',
            'Northern Cape': 'Mineral',
            'Mpumalanga': 'Mineral',
            'Limpopo': 'Mineral'
        }
        
        df['province'] = df['location'].map(province_map)
        df['economic_zone'] = df['province'].map(economic_zone_map)
        return df
    
    def get_loadshedding_status(self):
        """Get current loadshedding status from Eskom API"""
        try:
            url = "https://loadshedding.eskom.co.za/LoadShedding/GetStatus"
            response = requests.get(url)
            stage = int(response.text)
            return {
                'stage': stage,
                'status': f"Stage {stage}" if stage > 0 else "No loadshedding"
            }
        except:
            return {'stage': 0, 'status': 'Unknown'}