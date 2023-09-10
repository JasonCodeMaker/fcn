import torch
import torch.nn as nn
from torchvision.models import vgg16

class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # vgg16 backbone
        features = list(vgg16(pretrained=True).features.children())
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])

        # fcn layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, n_class, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        fc6 = self.fc6(pool5)
        fc7 = self.fc7(fc6)
        score_fr = self.score_fr(fc7)
        print(score_fr.shape)

        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        print(upscore2.shape, score_pool4.shape)
        score_sum = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(score_sum)
        score_pool3 = self.score_pool3(pool3)
        print(upscore_pool4.shape, score_pool3.shape)
        score_sum_final = upscore_pool4 + score_pool3

        upscore8 = self.upscore8(score_sum_final)
        return upscore8

if __name__ == '__main__':
    model = FCN8s()
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(input_tensor)
    print(output.shape)
