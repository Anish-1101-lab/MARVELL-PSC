import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class PhaseClassifier(nn.Module):
    def __init__(self, vocab_size=3000000, embed_dim=32, hidden_dim=64, num_phases=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_phases)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _out, (hn, _) = self.lstm(embedded)
        logits = self.fc(hn[-1])
        return logits

class ConditionedCacheModel(nn.Module):
    def __init__(self, vocab_size=3000000, num_phases=4, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.phase_embedding = nn.Embedding(num_phases, embed_dim)
        
        # Concat window embeddings (50 items) + phase embedding (1 item)
        self.fc1 = nn.Linear(embed_dim * 50 + embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Head 1: Tier Selection (4 classes: HBM, CXL, SSD, Cold)
        self.tier_head = nn.Linear(hidden_dim, 4)
        
        # Head 2: Prefetch Logic (Scalar output representing 0-8 blocks)
        self.prefetch_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_seq, phase):
        emb_seq = self.embedding(x_seq).view(x_seq.size(0), -1)
        emb_phase = self.phase_embedding(phase)
        
        combined = torch.cat([emb_seq, emb_phase], dim=1)
        out = self.relu(self.fc1(combined))
        
        tier_logits = self.tier_head(out)
        prefetch_val = self.prefetch_head(out) # Linear output for MSE
        
        return tier_logits, prefetch_val

def build_dataset_from_files(files_and_phases, window_size=50, sample_fraction=0.1):
    """
    files_and_phases: list of (path, phase_id)
    """
    X_seq, y_phase, y_tier, y_prefetch = [], [], [], []
    
    for path, phase_id in files_and_phases:
        print(f"Loading {path} as Phase {phase_id}...")
        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
            
        # Sample for speed if large
        if len(df) > 100000:
            df = df.sample(frac=sample_fraction).sort_index()

        trace = df["block_id"].values.astype(int)
        tiers = df["optimal_tier"].values.astype(int)
        
        for i in range(0, len(trace) - window_size, 5): # Stride for dataset density control
            X_seq.append(trace[i:i+window_size])
            target_idx = i + window_size
            y_phase.append(phase_id)
            y_tier.append(tiers[target_idx])
            
            # Prefetch target: 
            # If phase is 1 (Sequential) or 3 (BERT/NLP), prefetch high (e.g. 8)
            # If phase is 0 (Random), prefetch low (0)
            if phase_id in [1, 3]:
                y_prefetch.append(8.0)
            else:
                y_prefetch.append(0.0)
                
    return (torch.tensor(np.array(X_seq), dtype=torch.long), 
            torch.tensor(y_phase, dtype=torch.long), 
            torch.tensor(y_tier, dtype=torch.long), 
            torch.tensor(y_prefetch, dtype=torch.float32))

def train_phase_classifier(X_seq, y_phase, epochs=3, batch_size=256):
    print("\n--- Training 4-Phase Classifier (LSTM) ---")
    dataset = TensorDataset(X_seq, y_phase)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PhaseClassifier(num_phases=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for seq, tgt in loader:
            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits, tgt)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Acc: {correct/total:.4f}")
    return model

def train_dual_head_policy(X_seq, d_phases, y_tier, y_prefetch, epochs=5, batch_size=256):
    print("\n--- Training Dual-Head Policy (Tier CE + Prefetch MSE) ---")
    dataset = TensorDataset(X_seq, d_phases, y_tier, y_prefetch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ConditionedCacheModel(num_phases=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, p, t_tier, t_pref in loader:
            optimizer.zero_grad()
            p_tier, p_pref = model(seq, p)
            
            # Loss = 0.7 * CE(tier) + 0.3 * MSE(prefetch)
            loss_tier = ce_loss(p_tier, t_tier)
            loss_pref = mse_loss(p_pref.squeeze(), t_pref)
            loss = 0.7 * loss_tier + 0.3 * loss_pref
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Combined Loss: {total_loss/len(dataset):.4f}")
    return model

if __name__ == "__main__":
    train_files = [
        ("processed_traces/resnet_labeled.parquet", 0),
        ("dlio_mixed_labeled.csv", 1), # Sequential parts
        ("processed_traces/bert_labeled.parquet", 3)
    ]
    
    X_seq, y_phase, y_tier, y_prefetch = build_dataset_from_files(train_files)
    
    # 1. Train Phase Classifier
    phase_model = train_phase_classifier(X_seq, y_phase, epochs=5)
    torch.save(phase_model.state_dict(), "phase_classifier.pth")
    
    # 2. Predicted phases for conditioning
    phase_model.eval()
    with torch.no_grad():
        pred_phases = phase_model(X_seq).argmax(dim=1)
    
    # 3. Train Policy
    policy_model = train_dual_head_policy(X_seq, pred_phases, y_tier, y_prefetch, epochs=5)
    torch.save(policy_model.state_dict(), "policy_model_conditioned.pth")
    print("\nRetraining complete with 4 Tiers and 4 Phases.")
