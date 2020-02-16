import torch.nn as nn

class AstroNet(nn.Module):
    '''my net to classify'''

    def __init__(self):
        super(AstroNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2600, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        return self.net(x)
