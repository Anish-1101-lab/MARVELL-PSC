import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CacheTransformer(nn.Module):
    def __init__(self, vocab_size=100000, d_model=64, nhead=4, 
                 num_layers=3, num_tiers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Using feature dimension directly alongside embeddings or keeping it separate.
        # The instruction was: "A small transformer over the access sequence window."
        # x: (batch, seq_len) — sequence of block IDs
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_tiers)   # 3-way classification
    
    def forward(self, x):
        # x: (batch, seq_len) — sequence of block IDs
        emb = self.embed(x)                          # (batch, seq, d_model)
        out = self.transformer(emb)                  # (batch, seq, d_model)
        return self.head(out[:, -1, :])              # predict on last token

def train_slm(trace_sequence, labels, window_size=50, epochs=5):
    """
    Given a raw trace of block IDs and labeled dictionary, formats the data
    as sequences of length `window` and trains the CacheTransformer.
    """
    print(f"Preparing dataset with window size {window_size}...")
    X_seq = []
    y = []
    
    # We just need the block IDs for the transformer input as per instructions
    for idx in range(window_size, len(trace_sequence)):
        seq = trace_sequence[idx - window_size:idx]
        X_seq.append(seq)
        y.append(labels.get((trace_sequence[idx], idx), 2)) # 2 = cold tier default
        
    X_seq = torch.tensor(X_seq, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_seq, y)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = CacheTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Clip large block IDs to vocab size (modulo vocab_size) if necessary
            batch_X = batch_X % 100000
            
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        
    return model

if __name__ == "__main__":
    from week2_oracle import belady_3tier_labels
    
    # Generate random trace of length 2000 for local testing
    np.random.seed(42)
    dummy_trace = np.random.randint(0, 500, size=2000).tolist()
    labels = belady_3tier_labels(dummy_trace, hbm_size=10, ssd_size=20)
    
    model = train_slm(dummy_trace, labels, window_size=50, epochs=3)
    
    # Save the model
    torch.save(model.state_dict(), "cache_transformer.pth")
    print("Saved pretrained model to cache_transformer.pth")
