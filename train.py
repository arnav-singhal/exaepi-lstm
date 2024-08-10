"""Things to investigate (pretty much everything):
1. Different optimizers. Adam seems to outperform AdamW,
   but with normalized features this may cease to be true.
2. Optimizer parameters (pretty much all of them).
3. Some sort of annealing (PyTorch has some LR schedulers
   that seem promising).
4. Using torch.compile may speed up training. Not
   super optimistic about this, but a possibility."""

import torch
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time

loss_function = torch.nn.MSELoss()
def train(model, train_data, test_data, epochs=5000, lr=0.01, betas=(0.9, 0.999), log_interval = 1000):
    """!!! MUTATES _training_losses and _test_losses global variables !!!"""
    opt = optim.Adam(model.parameters(), lr=lr, betas=betas)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50)
    global _training_losses
    global _test_losses
    _training_losses = np.zeros(epochs)
    _test_losses = np.zeros(epochs)
    
    avg_window = np.min((log_interval // 2, 50))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.

        for data in train_data:
        # for i, data in enumerate(train_data):
            opt.zero_grad()
            past, future = data

            prediction = model(past)
            loss = loss_function(prediction, future)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # print(f"Batch {i} Loss: {loss}")

        # This is technically slightly nonuniform, batch_size does not
        # divide number of samples, providing more weight to the last
        # batch, but should be good enough for an idea of average loss
        total_loss /= len(train_data)

        model.eval()
        test_loss = 0.
        with torch.no_grad():
            for data in test_data:
                past, future = data
                prediction = model(past)
                test_loss += loss_function(prediction, future).item()

        test_loss /= len(test_data)

        _training_losses[epoch] = total_loss
        _test_losses[epoch] = test_loss

        # print(f"Epoch {epoch}")
        # print("Average train loss:", total_loss)
        # print("Average test loss:", test_loss)

        # scheduler.step(test_loss)
        if epoch % 1000 == 0:
            if epoch == 0:
                print(f"Epoch {epoch}")
                print("Average train loss:", total_loss)
                print("Average test loss:", test_loss)
                print('\n')
                start = time.time()
            else:
                end = time.time()
                print(f"Epoch {epoch}")
                print("Average train loss:", np.sum(_training_losses[epoch - 50 : epoch])/ 50)
                print("Average test loss:", np.sum(_test_losses[epoch - 50 : epoch])/ 50)
                print("Time:", end - start)
                # print("LR:", scheduler.get_last_lr())
                print('\n')
                start = time.time()
    return _training_losses, _test_losses

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_losses():
    """Plots test and train losses"""
    fig, ax = plt.subplots()
    t = np.nonzero(_training_losses == 0)[0][0] if 0 in _training_losses else len(_training_losses)

    ax.plot(_training_losses[:t], label="train error")
    ax.plot(moving_average(_training_losses[:t], 20), label="moving average")
    ax.plot(_test_losses[:t], label="test error")
    ax.plot(moving_average(_test_losses[:t], 20), label="moving average")
    ax.legend()
    return fig