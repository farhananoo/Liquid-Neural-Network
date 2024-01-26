import torch
import torch.nn.functional as f
from ncps.torch import LTC
from ncps.wirings import AutoNCP


class LiquidBaseline(torch.nn.Module):
    # See https://github.com/mlech26l/ncps
    def __init__(self):
        super(LiquidBaseline, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.rnn = LTC(16 * 62 * 62, AutoNCP(28, 2))

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x, _ = self.rnn(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
