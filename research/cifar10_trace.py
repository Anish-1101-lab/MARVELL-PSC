import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from week2_classifier_policy import PhaseClassifier, ConditionedCacheModel
from week6_eval import evaluate_metrics, evaluate_baseline
from week2_oracle import belady_3tier_labels

# Custom Dataset Wrapper to intercept and log block accesses
class TracedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_trace = []

    def __getitem__(self, index):
        # Log the index access (acting as a unique logical block_id for storage)
        self.access_trace.append(index)
        return super().__init__(index)
        
    def __getitem__(self, index):
        self.access_trace.append(index)
        img, target = self.data[index], self.targets[index]
        import torchvision.transforms.functional as F
        # barebones to avoid full transform overhead in dummy run
        return img, target

def run_cifar10_training():
    print("Downloading/Loading CIFAR-10...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # We only need a small subset to generate a fast trace for demo
    trainset = TracedCIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Batch size 32, run for 1 epoch just to get a healthy trace of a few thousand accesses
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Running PyTorch CIFAR-10 Training Epoch (Intercepting Data Accesses)...")
    
    for i, data in enumerate(trainloader, 0):
        if i >= 100: # Stop after 100 batches (3200 accesses) to keep demo quick
            break
            
    trace = trainset.access_trace
    print(f"Intercepted {len(trace)} real block accesses from DataLoader!")
    return trace

if __name__ == "__main__":
    # 1. Get real trace
    trace = run_cifar10_training()
    
    # 2. Add Ground Truth Labels using Oracle
    # We'll use Belady to figure out optimal tiers. 
    # Assume phase=0 (Training) for all accesses since this is a pure training loop
    print("Running Oracle on real CIFAR-10 trace...")
    oracle_mapping = belady_3tier_labels(trace, hbm_size=100, ssd_size=200)
    labels_dict = {}
    for t, blk in enumerate(trace):
        labels_dict[(blk, t)] = oracle_mapping[(blk, t)]
        
    # 3. Feed trace through our conditioned model
    print("Loading Trained Models...")
    try:
        phase_model = PhaseClassifier()
        phase_model.load_state_dict(torch.load("phase_classifier.pth"))
        phase_model.eval()
        
        policy_model = ConditionedCacheModel()
        policy_model.load_state_dict(torch.load("policy_model_conditioned.pth"))
        policy_model.eval()
        
        # Build sequence windows
        window_size = 50
        X_seq = []
        for idx in range(window_size, len(trace)):
            seq = trace[idx - window_size:idx]
            X_seq.append(seq)
            
        # Vocab size for CIFAR-10 tracing needs to be handled since block IDs map to indices 0-49999
        # Our model was trained with vocab_size=5000. 
        # Modulo the blocks to fit the vocab wrapper for this zero-shot generalization test
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.long) % 5000
        
        with torch.no_grad():
            preds_phase = phase_model(X_seq_tensor).argmax(dim=1)
            preds_policy = policy_model(X_seq_tensor, preds_phase).argmax(dim=1)
            
        actions = preds_policy.numpy().tolist()
        
        # 4. Report Hit Rate vs LRU
        model_hr, model_bw, model_p99 = evaluate_metrics(actions, labels_dict, trace)
        lru_hr = evaluate_baseline("LRU", trace, cache_size=100)
        
        print("\n=============================================")
        print(" CIFAR-10 REAL TRACE GENERALIZATION RESULTS ")
        print("=============================================")
        print(f"Model Zero-Shot Hit Rate : {model_hr*100:.2f}%")
        print(f"Baseline LRU Hit Rate    : {lru_hr*100:.2f}%")
        print("=============================================")
        print(f"Generalization Delta     : +{(model_hr - lru_hr)*100:.2f}%")
        
    except FileNotFoundError:
        print("No trained models found to evaluate.")
