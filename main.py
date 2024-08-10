import torch
import h5py
import time

import models
import train
import data

# "unique" file suffix to save plots and checkpoints without overwriting
FILE_SUFFIX = str(time.time())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA not available!\n")

print("Gathering data")
with h5py.File("../1k/county_data.h5", 'r') as f:
# with h5py.File("../1k/tract_data.h5", 'r') as f:
    infection_data = f['data'][()] / 1000
    infection_data = infection_data.reshape(*infection_data.shape[:-2], -1)
    # withdrawn_data = f['withdrawn'][()] / 1000
    # withdrawn_data = withdrawn_data.reshape(*withdrawn_data.shape[:-2], -1)

# infection_data = data.embed_withdrawn(infection_data, withdrawn_data)
# deltas = train.data_to_deltas(infection_data)
infection_data = data.embed_countdown(infection_data, sip_start=20, sip_length=30)


# training_loader, test_loader = data.get_loader(infection_data, device, in_days=50, batch_size=32)
# training_loader, test_loader = data.get_loader(deltas, device, in_days=50, batch_size=32)
training_loader, test_loader = data.get_lookback_loader(infection_data, device, lookback=25, batch_size=32)
print("Data loaded\n")

# mod = models.fifty_to_fifty(infection_data.shape[2], 64, num_layers=2)
# mod = models.one_to_fifty(infection_data.shape[2], 50, 64, num_layers=2) # 50 out days
mod = models.one_to_one(infection_data.shape[2], 64, num_layers=2)
mod = mod.to(device)

print("Training model")
training_losses, test_losses = train.train(mod, training_loader, test_loader, epochs=100, log_interval=10) # epochs should be O(5000) for normal data, O(10-100) for lookback data
print("Training complete\n")

loss_plot = train.plot_losses()

loss_plot.savefig("loss_plot_" + FILE_SUFFIX + ".png")

torch.save(mod.state_dict(), f"checkpoint_{infection_data.shape[2]}_{FILE_SUFFIX}.tar")
