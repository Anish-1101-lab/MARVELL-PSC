import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rocksdb_workload import RocksDBWorkloadGenerator
from predictor import LightGBMPredictor
from training_env import TraceTrainingEnv

def train():
    trace_path = '/tmp/rocksdb_trace_out/trace-human_readable_trace.txt'
    print(f"Loading RocksDB trace from {trace_path}...")
    wg = RocksDBWorkloadGenerator(trace_path)
    
    # We want to train on a good chunk
    df_full = wg.load_trace(max_rows=50000)
    
    train_len = min(40000, len(df_full))
    df_train_raw = df_full.iloc[:train_len].copy()
    
    # LightGBM Dataset
    df_train = wg.generate_dataset(df_train_raw, reuse_window=500)
    
    # Trace Arrays for the Gym Env
    trace, sizes = wg.get_eval_arrays(df_train_raw)
    
    print("Training LightGBM Predictor...")
    predictor = LightGBMPredictor()
    predictor.train(df_train)
    
    print("Initializing RL Training Environment...")
    env = TraceTrainingEnv(trace, sizes, predictor, alpha=10.0, beta=1.0, gamma=5.0)
    
    # Check if the environment is valid
    try:
        check_env(env, warn=True)
    except Exception as e:
        print(f"Gym Environment Check Warning: {e}")
    
    print("Instantiating PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, learning_rate=3e-4)
    
    print("Starting Training Loop (15,000 timesteps)...")
    # Quick training pass
    model.learn(total_timesteps=15000)
    
    print("Training Complete. Saving to ppo_migration_model.zip...")
    model.save("ppo_migration_model")

if __name__ == "__main__":
    train()
