import torch.nn as nn


class Model(nn.Module):
    """Ablation model without temporal convolution."""

    def __init__(self, num_classes: int = 101) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(3,64,kernel_size=(1, 7, 7),stride=(1, 2, 2),padding=(0, 3, 3))
        self.BN1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2),stride=(1, 2, 2),)

        self.conv2 = nn.Conv3d(
            64,
            128,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.BN2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
        )

        self.conv3 = nn.Conv3d(128,256, kernel_size=(1, 3, 3),padding=(0, 1, 1))
        self.BN3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(256,256,kernel_size=(1, 3, 3),padding=(0, 1, 1))
        self.BN4 = nn.BatchNorm3d(256)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=(2, 2, 2))

        self.conv5 = nn.Conv3d(256,512,kernel_size=(1, 3, 3),padding=(0, 1, 1))
        self.BN5 = nn.BatchNorm3d(512)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(512,512,kernel_size=(1, 3, 3),padding=(0, 1, 1))
        self.BN6 = nn.BatchNorm3d(512)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=(2, 2, 2))

        self.global_average_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.BN1(self.conv1(x))))
        x = self.pool2(self.relu2(self.BN2(self.conv2(x))))
        x = self.relu3(self.BN3(self.conv3(x)))
        x = self.relu4(self.BN4(self.conv4(x)))
        x = self.pool3(x)
        x = self.relu5(self.BN5(self.conv5(x)))
        x = self.relu6(self.BN6(self.conv6(x)))
        x = self.pool4(x)
        x = self.global_average_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        return self.fc(x)
