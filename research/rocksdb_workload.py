import pandas as pd
import numpy as np
from typing import Dict

class RocksDBWorkloadGenerator:
    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        
        # We only care about block_id, block_size, block_type
        # trace_analyzer outputs: `key query_type_id cf_id value_size time_in_micorsec`
        # separated by spaces
        self.columns = [
            "key", "query_type", "cf_id", "value_size", "timestamp"
        ]
        
    def load_trace(self, max_rows=None) -> pd.DataFrame:
        """Loads the trace and maps block_id to contiguous integers to match simulator format."""
        print(f"Loading RocksDB trace from {self.trace_file}...")
        df = pd.read_csv(self.trace_file, names=self.columns, sep=' ', nrows=max_rows)
        
        # Map key (hex string) to a contiguous integer space starting from 1
        unique_blocks = df['key'].unique()
        block_id_map = {old: new + 1 for new, old in enumerate(unique_blocks)}
        
        # Convert the trace to sequential object accesses
        df['obj_id'] = df['key'].map(block_id_map)
        
        # Convert value_size to units (e.g., 100 bytes is ~1 unit here for baseline simulation)
        # We fill NaN or missing with 100 bytes (typical test setting)
        df['value_size'] = df['value_size'].fillna(100)
        df['size_units'] = np.maximum(1, np.ceil(df['value_size'] / 4096.0)).astype(int)
        
        return df
        
    def generate_dataset(self, df: pd.DataFrame, reuse_window: int = 2000) -> pd.DataFrame:
        """
        Calculates simulator-specific features: recency, frequency, time_to_next_access, reuse_probability.
        """
        print(f"Parsing RocksDB trace for dataset features ({len(df)} accesses)...")
        num_accesses = len(df)
        trace_obj_ids = df['obj_id'].values
        trace_sizes = df['size_units'].values
        
        last_seen: Dict[int, int] = {}
        frequency: Dict[int, int] = {}
        
        records = []
        
        # Forward pass for historical features
        for t in range(num_accesses):
            obj_id = trace_obj_ids[t]
            size = trace_sizes[t]
            
            recency = t - last_seen.get(obj_id, -1) if obj_id in last_seen else -1
            freq = frequency.get(obj_id, 0)
            
            records.append({
                'time_step': t,
                'obj_id': obj_id,
                'size': size,
                'recency': recency,
                'frequency': freq
            })
            
            last_seen[obj_id] = t
            frequency[obj_id] = freq + 1
            
        dataset = pd.DataFrame(records)
        
        # Backward pass for future targets
        next_seen: Dict[int, int] = {}
        t_i = np.full(num_accesses, -1)
        p_i = np.zeros(num_accesses)
        
        for t in range(num_accesses - 1, -1, -1):
            obj_id = trace_obj_ids[t]
            if obj_id in next_seen:
                t_next = next_seen[obj_id]
                time_to_next = t_next - t
                t_i[t] = time_to_next
                if time_to_next <= reuse_window:
                    p_i[t] = 1.0
            next_seen[obj_id] = t
            
        dataset['time_to_next_access'] = t_i
        dataset['reuse_probability'] = p_i
        
        return dataset
        
    def get_eval_arrays(self, df: pd.DataFrame):
        """Returns the trace array and sizes dictionary/array for the engine."""
        trace = df['obj_id'].values
        # Sizes per object ID must be stable. Since we mapped IDs sequentially, we can create an array.
        max_obj_id = df['obj_id'].max()
        sizes = np.ones(max_obj_id + 1, dtype=int)
        
        # Group by obj_id and take max size_units (usually size is constant per block)
        obj_sizes = df.groupby('obj_id')['size_units'].max()
        sizes[obj_sizes.index] = obj_sizes.values
        
        return trace, sizes
