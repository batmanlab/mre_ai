# Simple arch for debug
import torch
import torch.nn as nn


class Debug(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Debug, self).__init__()

        self.conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        x = self.conv_1x1(input)
        return x


if __name__ == "__main__":
    model = Debug(in_channels=9, out_channels=1)
    input = torch.rand(1, 9, 32, 256, 256)
    output = model(input)
    print(output.size())
