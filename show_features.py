import torch

import hmax
from dataloaders import train_dataloader
from matplotlib import pyplot as plt
print('Extract HMAX feature')
model = hmax.HMAX('./hmax/universal_patch_set.mat')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

for images, labels in train_dataloader:
    s1, c1, s2, c2 = model.run_all_layers(images)
    plt.imshow(images[0,0], cmap='gray')
    plt.show()
    for s1_feature in s1:
        plt.imshow(s1_feature[0, 0], cmap='gray')
        plt.show()


