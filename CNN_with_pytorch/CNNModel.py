from torch import nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.CNN_arct = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(30, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
        )
        self.norm = nn.Dropout()
        self.lin = nn.Linear(100, 10)

    def forward(self, x):
        x = self.CNN_arct(x)
        x = x.view(-1, x.size(1))
        x = self.norm(x)
        return self.lin(x)
