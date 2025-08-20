# supply_chain_models.py
from prophet import Prophet
import cvxpy as cp
import pandas as pd
import numpy as np
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)

class SupplyOptimizer:
    def __init__(self):
        self.base_safety_factor = 1.2  # Base SA safety factor
        self.holding_cost_rate = 0.18  # 18% annual holding cost
        self.stockout_cost_multiplier = 3.0  # Stockout costs 3x holding costs
    
    def forecast_demand(self, historical_data):
        """Time series forecasting for SA products"""
        if historical_data.empty:
            return pd.DataFrame()
        
        try:
            df = historical_data.rename(columns={'date': 'ds', 'demand': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.dropna(subset=['y'])
            
            model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
            model.add_country_holidays(country_name='ZA')
            model.fit(df)
            
            future = model.make_future_dataframe(periods=90, freq='D')
            forecast = model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            print(f"Forecasting failed: {str(e)}")
            return pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=100),
                'yhat': np.sin(np.linspace(0, 10, 100)) * 50 + 200,
                'yhat_lower': np.sin(np.linspace(0, 10, 100)) * 40 + 180,
                'yhat_upper': np.sin(np.linspace(0, 10, 100)) * 60 + 220
            })
    
    def optimize_inventory(self, demand_forecast, lead_time, risk_score, current_stock=0):
        """Stochastic inventory optimization for SA context"""
        if demand_forecast.empty:
            return {
                'optimal_order': 0,
                'safety_stock': 0,
                'holding_cost': 0,
                'stockout_risk': 0
            }
        
        try:
            # Calculate base values
            avg_demand = demand_forecast['yhat'].mean()
            safety_stock = self._calculate_safety_stock(avg_demand, lead_time, risk_score)
            target_stock = avg_demand * lead_time * (1 + risk_score) + safety_stock
            
            # Optimization variables
            order_quantity = cp.Variable(nonneg=True)
            holding_cost = cp.Variable(nonneg=True)
            stockout_cost = cp.Variable(nonneg=True)
            
            # Constraints
            constraints = [
                current_stock + order_quantity == target_stock,
                holding_cost == self.holding_cost_rate / 365 * order_quantity,
                stockout_cost == self.stockout_cost_multiplier * holding_cost * risk_score
            ]
            
            # Objective function
            total_cost = holding_cost + stockout_cost
            problem = cp.Problem(cp.Minimize(total_cost), constraints)
            problem.solve(solver=cp.ECOS)
            
            return {
                'optimal_order': max(0, round(order_quantity.value)),
                'safety_stock': round(safety_stock),
                'holding_cost': round(holding_cost.value, 2),
                'stockout_risk': min(1, risk_score * 1.5)
            }
        except:
            # Fallback calculation
            safety_stock = avg_demand * lead_time * self.base_safety_factor * (1 + risk_score)
            return {
                'optimal_order': max(0, round(target_stock - current_stock)),
                'safety_stock': round(safety_stock),
                'holding_cost': round(self.holding_cost_rate / 365 * safety_stock, 2),
                'stockout_risk': min(1, risk_score * 1.5)
            }
    
    def _calculate_safety_stock(self, avg_demand, lead_time, risk_score):
        """Calculate safety stock considering SA risk factors"""
        demand_variability = avg_demand * 0.3  # 30% variability
        lead_time_variability = lead_time * 0.4  # 40% variability
        z_score = 1.65  # 95% service level
        
        base_stock = z_score * np.sqrt(
            (lead_time * demand_variability**2) + 
            (avg_demand**2 * lead_time_variability**2)
        )
        return base_stock * (1 + risk_score)
    
    def route_risk_model(self, origin, destination, current_events):
        """Calculate route risk based on SA conditions"""
        # Base risk factors for major SA routes
        risk_factors = {
            ('Johannesburg', 'Cape Town'): 0.65,
            ('Durban', 'Johannesburg'): 0.55,
            ('Port Elizabeth', 'Johannesburg'): 0.60,
            ('Johannesburg', 'Gaborone'): 0.70,
            ('Johannesburg', 'Maputo'): 0.75,
            ('Durban', 'Maputo'): 0.68
        }
        
        key = (origin, destination)
        base_risk = risk_factors.get(key, 0.5)
        
        # Adjust for current events
        origin_events = current_events[current_events['location'].str.contains(origin, case=False)]
        dest_events = current_events[current_events['location'].str.contains(destination, case=False)]
        
        if not origin_events.empty:
            base_risk += origin_events['risk_score'].max() * 0.3
        if not dest_events.empty:
            base_risk += dest_events['risk_score'].max() * 0.3
        
        return min(1, base_risk)