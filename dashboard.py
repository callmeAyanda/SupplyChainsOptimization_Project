# dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_processor import SADataProcessor
from event_detector import ZAEventDetector
from supply_chain_models import SupplyOptimizer
from datetime import datetime, timedelta

def safe_df_access(df, column, default=0):
    """Safely access DataFrame columns with fallback values"""
    try:
        if df.empty or column not in df.columns:
            return default
        return df[column]
    except:
        return default

# Initialize components
data_processor = SADataProcessor()
event_detector = ZAEventDetector()
optimizer = SupplyOptimizer()

# Set up page
st.set_page_config(
    page_title="SA Supply Chain Optimizer",
    page_icon="🇿🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetricLabel { font-weight: bold !important; }
    .st-bb { background-color: #000000; }
    .st-at { background-color: #000000; }
    .css-18e3th9 { padding-top: 2rem; padding-bottom: 2rem; }
    .header-style { color: #1f77b4; border-bottom: 2px solid #1f77b4; }
    .risk-high { color: #d62728 !important; font-weight: bold; }
    .risk-medium { color: #ff7f0e !important; }
    .risk-low { color: #2ca02c !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Flag_of_South_Africa.svg/1200px-Flag_of_South_Africa.svg.png", 
             width=100)
    st.title("SA Supply Chain Optimizer")
    
    st.header("Configuration")
    product = st.selectbox("Product Category", 
                          ["Mining Equipment", "Agricultural Produce", "Manufactured Goods", "Pharmaceuticals"])
    origin = st.selectbox("Origin", 
                         ["Johannesburg", "Durban", "Cape Town", "Port Elizabeth", "East London"])
    destination = st.selectbox("Destination", 
                             ["Johannesburg", "Cape Town", "Durban", "Gaborone (BW)", "Maputo (MZ)"])
    lead_time = st.slider("Lead Time (days)", 1, 30, 14)
    current_stock = st.number_input("Current Stock Level", min_value=0, value=1000)
    
    st.divider()
    st.header("Real-time Status")
    
    # Load shedding status
    loadshedding = data_processor.get_loadshedding_status()
    st.metric("Loadshedding Stage", 
             f"Stage {loadshedding['stage']}" if loadshedding['stage'] > 0 else "None",
             delta=None if loadshedding['stage'] == 0 else f"In progress" if loadshedding['stage'] > 0 else "No loadshedding",
             delta_color="inverse")
    
    # Currency rates
    st.metric("USD/ZAR", "18.75", "-0.2%")

# Data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_all_data():
    try:
        freight_data = data_processor.load_freight_data()
        commodity_prices = data_processor.get_commodity_prices()
        news_data = data_processor.get_sa_news()
        processed_events = event_detector.process_news(news_data)
        risk_scores = event_detector.generate_risk_score(processed_events)
        
        # Add geo-coordinates
        if not risk_scores.empty:
            locations = data_processor.geocode_sa_locations(risk_scores['location'].tolist())
            risk_scores['lat'] = risk_scores['location'].map(lambda x: locations.get(x, (None, None))[0])
            risk_scores['lon'] = risk_scores['location'].map(lambda x: locations.get(x, (None, None))[1])
        else:
            # Create empty columns if no risk data
            risk_scores['lat'] = None
            risk_scores['lon'] = None

        # Generate sample demand data
        demand_data = pd.DataFrame({
            'date': pd.date_range(end=datetime.today(), periods=365),
            'demand': np.abs(np.sin(np.linspace(0, 10, 365)) * 50 + 100 + np.random.normal(0, 10, 365))
        })
        
        return {
            'freight': freight_data,
            'commodities': commodity_prices,
            'news': news_data,
            'events': processed_events,
            'risk': risk_scores,
            'demand': demand_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty data structures on error
        return {
            'freight': pd.DataFrame(),
            'commodities': pd.DataFrame(),
            'news': pd.DataFrame(),
            'events': pd.DataFrame(),
            'risk': pd.DataFrame(columns=['location', 'risk_score', 'latest_event', 'event_count', 'lat', 'lon']),
            'demand': pd.DataFrame()
        }

data = load_all_data()

# Dashboard layout
st.title("🇿🇦 South African Supply Chain Intelligence Platform")
st.caption("Real-time risk monitoring and inventory optimization for SA logistics")

# Top KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
route_risk = optimizer.route_risk_model(origin, destination, data['risk']) if not data['risk'].empty else 0.5
kpi1.metric("Route Risk", f"{route_risk*100:.0f}%", 
           "High Risk" if route_risk > 0.7 else "Medium Risk" if route_risk > 0.4 else "Low Risk",
           delta_color="inverse" if route_risk > 0.5 else "normal")

avg_delay = data['freight']['delay_hours'].mean() if not data['freight'].empty and 'delay_hours' in data['freight'].columns else 8.5
kpi2.metric("Avg Transit Delay", f"{avg_delay:.1f} hours", 
           "Improving" if avg_delay < 10 else "Worsening", 
           delta_color="inverse" if avg_delay > 12 else "normal")

if not data['commodities'].empty and 'change_pct' in data['commodities'].columns:
    commodity_risk = data['commodities']['change_pct'].abs().mean()
else:
    commodity_risk = 1.2
kpi3.metric("Commodity Volatility", f"{commodity_risk:.1f}%", 
           "High Volatility" if commodity_risk > 1.5 else "Stable")

if not data['risk'].empty and 'risk_score' in data['risk'].columns:
    active_events = len(data['risk'][data['risk']['risk_score'] > 0.3])
else:
    active_events = 0
kpi4.metric("Active Disruptions", active_events, 
           "Critical" if active_events > 5 else "Normal" if active_events > 0 else "None",
           delta_color="inverse" if active_events > 3 else "normal")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Risk Intelligence", "Inventory Optimization", "Route Analysis"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-time Disruption Map", divider='blue')
        if not data['risk'].empty:
            fig = px.scatter_mapbox(
                data['risk'], 
                lat='lat', 
                lon='lon', 
                size='risk_score',
                color='risk_score',
                hover_name='location',
                hover_data={'risk_score': ':.2f', 'latest_event': True, 'event_count': True},
                color_continuous_scale=px.colors.sequential.Redor,
                zoom=5,
                height=500,
                center=dict(lat=-28.5, lon=25)
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No current disruption data available")
    
    with col2:
        st.subheader("Active Disruptions", divider='blue')
        if not data['risk'].empty:
            high_risk = data['risk'][data['risk_score'] > 0.5]
            med_risk = data['risk'][(data['risk_score'] > 0.3) & (data['risk_score'] <= 0.5)]
            
            if not high_risk.empty:
                st.markdown("#### 🔴 High Risk Events")
                for _, row in high_risk.iterrows():
                    st.markdown(f"**{row['location']}** - Risk: {row['risk_score']:.2f}  \n"
                               f"Last event: {row['latest_event'].strftime('%Y-%m-%d')} | Events: {row['event_count']}")
            
            if not med_risk.empty:
                st.markdown("#### 🟠 Medium Risk Events")
                for _, row in med_risk.iterrows():
                    st.markdown(f"**{row['location']}** - Risk: {row['risk_score']:.2f}  \n"
                               f"Last event: {row['latest_event'].strftime('%Y-%m-%d')} | Events: {row['event_count']}")
            
            if high_risk.empty and med_risk.empty:
                st.success("No significant disruptions detected")
        else:
            st.info("No current disruption data available")
        
        st.subheader("Commodity Prices", divider='blue')
        if not data['commodities'].empty:
            for _, row in data['commodities'].iterrows():
                delta = f"{row['change_pct']:.1f}%"
                st.metric(row['commodity'], f"{row['price']} {row['unit']}", delta)
        else:
            st.info("Commodity data unavailable")

with tab2:
    st.subheader("Inventory Optimization", divider='blue')
    
    # Get demand forecast
    forecast = optimizer.forecast_demand(data['demand'])
    
    if not forecast.empty:
        # Get optimization results
        opt_result = optimizer.optimize_inventory(
            forecast, lead_time, route_risk, current_stock
        )
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Optimal Order Quantity", f"{opt_result['optimal_order']:,} units", 
                   help="Recommended purchase order based on current risk")
        col2.metric("Safety Stock Level", f"{opt_result['safety_stock']:,} units", 
                   "Risk-adjusted buffer stock")
        col3.metric("Daily Holding Cost", f"R{opt_result['holding_cost']:,.2f}", 
                   "Estimated inventory carrying cost")
        col4.metric("Stockout Risk", f"{opt_result['stockout_risk']*100:.1f}%", 
                   "High risk" if opt_result['stockout_risk'] > 0.3 else "Acceptable",
                   delta_color="inverse" if opt_result['stockout_risk'] > 0.3 else "normal")
        
        # Forecast visualization
        st.subheader("Demand Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(31,119,180,0.2)'),
            name='Upper Bound'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(31,119,180,0.2)'),
            name='Lower Bound'
        ))
        # Add a vertical line using a different approach that avoids the timestamp arithmetic issue
        today = datetime.today().date()
        today_str = today.strftime('%Y-%m-%d')
        # Create a separate trace for the vertical line
        fig.add_trace(go.Scatter(
            x=[today_str, today_str],
            y=[forecast['yhat'].min(), forecast['yhat'].max()],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Today',
            showlegend=False
        ))
        # Add annotation for the vertical line
        fig.add_annotation(
            x=today_str,
            y=forecast['yhat'].max(),
            text="Today",
            showarrow=False,
            yshift=10,
            font=dict(color='red')
        )

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Demand (units)',
            hovermode="x unified",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.subheader("Recommendations")
        if route_risk > 0.6:
            st.warning("**High-Risk Route Detected**")
            st.markdown("""
            - Diversify suppliers to alternative regions
            - Increase safety stock by 20-30%
            - Consider air freight for critical components
            - Activate contingency logistics partners
            """)
        else:
            st.success("**Normal Operations**")
            st.markdown("""
            - Maintain standard safety stock levels
            - Monitor key risk indicators daily
            - Review supplier performance quarterly
            """)
    else:
        st.error("Demand forecasting not available")

with tab3:
    st.subheader(f"Route Analysis: {origin} → {destination}", divider='blue')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Route statistics
        st.markdown("#### Route Metrics")
        
        # Get route-specific data
        route_data = data['freight'][
            (data['freight']['origin'].str.contains(origin)) & 
            (data['freight']['destination'].str.contains(destination))
        ]
        
        if not route_data.empty:
            avg_delay = route_data['delay_hours'].mean()
            reliability = (route_data['delay_hours'] < 24).mean() * 100
            st.metric("Average Delay", f"{avg_delay:.1f} hours")
            st.metric("On-time Reliability", f"{reliability:.0f}%")
            
            # Delay distribution
            st.markdown("#### Delay Distribution")
            fig = px.histogram(
                route_data, 
                x='delay_hours', 
                nbins=20,
                labels={'delay_hours': 'Delay (hours)'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data for this route")
        
    with col2:
        # Risk factors
        st.markdown("#### Risk Factors")
        
        # Breakdown of risk contributors
        risk_factors = {
            'Infrastructure': 0.3 * route_risk,
            'Weather': 0.2 * route_risk,
            'Labor Issues': 0.25 * route_risk,
            'Border Delays': 0.15 * route_risk,
            'Security': 0.1 * route_risk
        }
        
        fig = go.Figure(go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            marker_color=['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
        ))
        fig.update_layout(
            title="Risk Contribution",
            xaxis_title="Risk Score",
            yaxis_title="Factor",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Alternative routes
        st.markdown("#### Alternative Routes")
        alternatives = {
            ('Johannesburg', 'Cape Town'): [('Via N1 Direct', 0.60), ('Via Bloemfontein', 0.55)],
            ('Durban', 'Johannesburg'): [('Via N3 Direct', 0.55), ('Via Pietermaritzburg', 0.50)],
            ('Port Elizabeth', 'Johannesburg'): [('Via N10', 0.65), ('Via Bloemfontein', 0.62)]
        }
        
        if (origin, destination) in alternatives:
            for alt in alternatives[(origin, destination)]:
                st.markdown(f"**{alt[0]}**  \nRisk: {alt[1]*100:.0f}%")
        else:
            st.info("No alternative routes available")

# Footer
st.divider()
st.caption("© 2024 South African Supply Chain Intelligence Platform | Data Sources: Transnet, News24, Eskom, JSE")
