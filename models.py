"""Things to investigate:
1. Training the one_to_fifty model with normalized features
        - especially with shelter-in-place data embedded
2. Seeing effect of using tract-level vs county-level data
3. Seeing effect of including/not including withdrawn numbers,
   rather than just days of shelter-in-place remaining.
4. Predicting future total infection counts, rather than 
   stratifying output by county/age group."""

import torch
import torch.nn as nn

class fifty_to_fifty(nn.Module):
    """This model is the first try—it's inherently misspecified, as the n-th predicted
    day is only dependent on the first n input days. However, it's the one that got the
    best training and results so here for comparison."""
    def __init__(self, num_features, hidden_size, num_layers=1):
        super(fifty_to_fifty, self).__init__()
        self.lstm = nn.LSTM(num_features,
                            hidden_size, batch_first=True, num_layers=num_layers)
        self.lin = nn.Linear(hidden_size, num_features)
        # self.relu = nn.ReLU()

    def forward(self, past):
        out, _ = self.lstm(past)
        # out = self.relu(out)
        out = self.lin(out)
        return out

class one_to_fifty(nn.Module):
    """This model is the correct version of fifty_to_fifty, predicting the next out_days
    number of days from the last input day's hidden state, but didn't train correctly
    due to unnormalized features. Likely the most promising model if trained post-normalization"""
    def __init__(self, num_features, out_days, hidden_size, num_layers=1):
        super(one_to_fifty, self).__init__()
        self.lstm = nn.LSTM(num_features,
                            hidden_size, batch_first=True, num_layers=num_layers)
        self.lin = nn.Linear(hidden_size, out_days * num_features)
        # self.relu = nn.ReLU()
        self.out_days = out_days
        self.num_features = num_features

    def forward(self, past):
        out, _ = self.lstm(past)
        # out = self.relu(out[..., -1, :])

        out = out[..., -1, :]
        # Shape: [batch_size, hidden_size]

        out = self.lin(out)
        # Shape: [batch_size, out_days * counties * agegroups]

        return torch.unflatten(out, -1, (self.out_days, self.num_features))

class one_to_one(nn.Module):
    """Predicts one day in the future based on the last input day's hidden state.
    Able to be chained to predict farther into the future, but this will propogate
    errors such that it would seem that one_to_fifty is preferable."""
    def __init__(self, num_features, hidden_size, num_layers):
        super(one_to_one, self).__init__()
        self.lstm = nn.LSTM(num_features, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_size, num_features)

    def forward(self, past):
        out, _ = self.lstm(past)
        out = out[..., -1, :]
        out = self.lin(out)
        return out