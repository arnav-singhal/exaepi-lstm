"""Things to investigate:
1. Shuffling before assigning train/test split
2. Finding optimal batch_size (I belive this amounts
   to seeing what the biggest batch_size that doesn't
   cause memory problems is)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def embed_countdown(data, sip_start=20, sip_length=30):
    countdown = np.hstack((np.zeros(sip_start), sip_length - np.arange(sip_length), np.zeros(data.shape[1] - sip_start - sip_length))) / sip_length
    return np.dstack((data, np.tile(countdown, (data.shape[0], 1))))

def embed_withdrawn(data, withdrawn_data):
    return np.dstack((data, withdrawn_data))

def get_loader(data, device, in_days=50, batch_size=32):
    cutoff = data.shape[0] - data.shape[0] // 10
    
    train_in = torch.tensor(data[:cutoff, :in_days], device=device).flatten(start_dim=2, end_dim=3)
    train_out = torch.tensor(data[:cutoff, in_days:], device=device).flatten(start_dim=2, end_dim=3)
    test_in = torch.tensor(data[cutoff:, :in_days], device=device).flatten(start_dim=2, end_dim=3)
    test_out = torch.tensor(data[cutoff:, in_days:], device=device).flatten(start_dim=2, end_dim=3)
    return (DataLoader(TensorDataset(train_in, train_out), shuffle=True, batch_size=32),
            DataLoader(TensorDataset(test_in, test_out), shuffle=True, batch_size=32))

def get_lookback_loader(data, device, lookback=25, batch_size=128):
    train_num = data.shape[0] // 10
    cutoff = data.shape[0] - train_num
    days = data.shape[1]
    features = data.shape[2]
    
    train_in = np.zeros((cutoff * (days - lookback), lookback, features), dtype=np.float32)
    train_out = np.zeros((cutoff * (days - lookback), features), dtype=np.float32)
    test_in = np.zeros((train_num * (days - lookback), lookback, features), dtype=np.float32)
    test_out = np.zeros((train_num * (days - lookback), features), dtype=np.float32)

    for i in range(lookback, 100):
        train_in[cutoff * (i - lookback):cutoff * (i - lookback + 1)] = data[:cutoff, i-lookback:i]
        train_out[cutoff * (i - lookback):cutoff * (i - lookback + 1)] = data[:cutoff, i]
        test_in[train_num * (i - lookback):train_num * (i - lookback + 1)] = data[cutoff:, i-lookback:i]
        test_out[train_num * (i - lookback):train_num * (i - lookback + 1)] = data[cutoff:, i]

    train_in = torch.tensor(train_in, device=device)
    train_out = torch.tensor(train_out, device=device)
    test_in = torch.tensor(test_in, device=device)
    test_out = torch.tensor(test_out, device=device)

    return (DataLoader(TensorDataset(train_in, train_out), shuffle=True, batch_size=batch_size),
            DataLoader(TensorDataset(test_in, test_out), shuffle=True, batch_size=batch_size))

def data_to_deltas(data):
    """data -> percent change to next day
    works for both flattened and unflattened data"""
    offset = np.zeros(data.shape, dtype=np.float32)
    offset[:, :-1] = data[:, 1:]
    return (data - offset) / data + 1

def deltas_to_data(deltas):
    out = np.zeros(deltas.shape, dtype=np.float32)
    out[0] = deltas[0] / (1 - deltas[0])
    for i in range(1, len(deltas)):
        out[i] = (deltas[i] + out[i - 1]) / (1 - deltas[i])
    return out