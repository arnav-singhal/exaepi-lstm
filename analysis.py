import torch
import os

import data
import models

#hard-coded...
LOOKBACK = 25
TOTAL_DAYS = 100

checkpoint_file = "checkpoint_46_1723245959.5877738.tar"

num_features = int(checkpoint_file.split('_')[1])
time_stamp = checkpoint_file.split('_')[1][:-4]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA not available!\n")

# mod = models.fifty_to_fifty(num_features, 64, num_layers=2)
# mod = models.one_to_fifty(num_features, 50, 64, num_layers=2) # 50 out days
mod = models.one_to_one(num_features, 64, num_layers=2).to(device)

mod.load_state_dict(torch.load(checkpoint_file))

def full(model, first, total_days):
    """recursively generate forward.
    only use for one_to_one"""
    out = torch.zeros((total_days, first.shape[1]), device=device)
    out[:first.shape[0]] = first
    for i in range(25, 100):
        out[i] = model(out[i - 25:i])
    return out

with h5py.File("../1k/county_data.h5", 'r') as f:
# with h5py.File("../1k/tract_data.h5", 'r') as f:
    infection_data = f['data'][()]
    infection_data = infection_data.reshape(*infection_data.shape[:-2], -1)
    num_bins = infection_data.shape[2]
    # withdrawn_data = f['withdrawn'][()]
    # withdrawn_data = withdrawn_data.reshape(*withdrawn_data.shape[:-2], -1)

# infection_data = data.embed_withdrawn(infection_data, withdrawn_data)
# infection_data = train.data_to_deltas(infection_data)
infection_data = data.embed_countdown(infection_data, sip_start=20, sip_length=30)


cutoff = infection_data.shape[0] - infection_data.shape[0] // 10
test_data = infection_data[cutoff:]

os.makedirs(f"{time_stamp}/", exist_ok=True)
for n, datum in test_data:

    # curve = model(datum[:LOOKBACK])[:, num_bins]
    curve = full(model, datum[:LOOKBACK], TOTAL_DAYS)[:, :num_bins]
    
    # Use if trained on deltas instead of raw counts
    curve = data.deltas_to_data(curve)

    plt.plot(curve.detach().cpu().numpy().sum(axis=1), label="prediction")
    plt.plot(datum[:, num_bins].sum(axis=1), "--", label="truth")
    plt.legend()
    plt.title("Infection counts over time")
    plt.xlabel("Day")
    plt.ylabel("Infections (in thousands)")
    plt.savefig(f"{time_stamp}/test_{n:02}.png")
