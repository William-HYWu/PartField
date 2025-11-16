import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.stats as stats
from usdf.models import mlp

device = "cuda"
dtype = torch.float32

# Generate some fake data.
in_1 = torch.linspace(0.0, 1.0, 50, device=device, dtype=dtype).unsqueeze(0)
in_2 = torch.linspace(-1.0, -0.2, 50, device=device, dtype=dtype).unsqueeze(0)
a = torch.cat([in_1, in_2], dim=0)

# Create the mean/variance that we want to optimize.
# mean = torch.tensor(np.random.random(), device=device, dtype=dtype, requires_grad=True)
# var = torch.tensor(np.random.random(), device=device, dtype=dtype, requires_grad=True)

# Create little model to predict parameters.
model = mlp.build_mlp(1, 2, hidden_sizes=[8, 8], device=device)

# Create adam optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop.
for i in range(1000):
    # Compute loss.
    out = model(torch.tensor([[0.0], [1.0]]).to(device))
    mean = out[:, 0]
    var = out[:, 1]
    var = torch.exp(var)
    loss = F.gaussian_nll_loss(mean.flatten().repeat_interleave(a.shape[1]), a.flatten(),
                               var.flatten().repeat_interleave(a.shape[1]))

    # Backprop.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss.
    if i % 10 == 0:
        print("loss: ", loss.item())

    fig, ax = plt.subplots(2)
    for idx in range(2):
        ax[idx].scatter(a.cpu().numpy()[idx], np.zeros(a.shape[1]),
                        label="Ground truth SDF values")
        x = np.linspace(a[idx].min().item() - 0.05, a[idx].max().item() + 0.05, 100)
        ax[idx].plot(x, stats.norm.pdf(x, mean[idx].item(), np.sqrt(var[idx].item())))
        ax[idx].plot(x, stats.norm.pdf(x, np.mean(a[idx].cpu().numpy()), np.std(a[idx].cpu().numpy())))
    plt.savefig("out/train_gaussian_animate/{}.png".format(i))
    # plt.show()
    plt.close()
