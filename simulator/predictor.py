import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class LightGBMPredictor:
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.clf_reuse = lgb.LGBMClassifier(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            random_state=random_state,
            verbose=-1
        )
        self.reg_time = lgb.LGBMRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            random_state=random_state,
            verbose=-1
        )
        self.is_trained = False
        self.features = ['size', 'recency', 'frequency']
        
    def train(self, df: pd.DataFrame):
        """
        Trains models on historical access data.
        df must contain features and targets ('reuse_probability', 'time_to_next_access').
        We drop rows where time_to_next_access == -1 (end of trace).
        """
        train_df = df[df['time_to_next_access'] != -1].copy()
        
        if len(train_df) == 0:
            print("Warning: Not enough data to train LightGBM.")
            return

        X = train_df[self.features]
        y_clf = train_df['reuse_probability']
        y_reg = train_df['time_to_next_access']
        
        # Train classifier for reuse probability
        self.clf_reuse.fit(X, y_clf)
        
        # Train regressor for time-to-next-access
        self.reg_time.fit(X, y_reg)
        
        self.is_trained = True
        
    def predict(self, size: int, recency: int, frequency: int) -> Tuple[float, float]:
        """
        Given the current state of an object, predicts:
        P_i (reuse probability)
        T_i (time-to-next-access)
        """
        if not self.is_trained:
            # Fallback values if predictor uninitialized
            return (0.0, float('inf'))
            
        x_in = pd.DataFrame([{'size': size, 'recency': recency, 'frequency': frequency}])
        
        # Predict probability of reuse (class 1)
        # Note: If predict_proba fails (e.g., only one class in training set), handle gracefully
        try:
            p_i = self.clf_reuse.predict_proba(x_in)[0, 1]
        except IndexError:
            # If the model only learned 1 class
            p_i = float(self.clf_reuse.predict(x_in)[0])
            
        # Predict time to next access
        t_i = self.reg_time.predict(x_in)[0]
        
        # Sanity bounds (t_i shouldn't be negative)
        t_i = max(0.0, t_i)
            
        return p_i, t_i
