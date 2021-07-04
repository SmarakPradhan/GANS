import torch
import torch.nn as nn

# Discriminator and Generator implementation from DCGAN paper




class Discriminator(nn.Module):
    def __init__(self,img_channels,disc_features):
        super(Discriminator,self).__init__()
        #Input shape = N x img_channels x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels,disc_features,kernel_size=4, stride=2, padding=1),
            # 32 x 32
            nn.LeakyReLU(0.2),
            # block(in_channels, out_channels, kernel_size, stride, padding)
            self.block(disc_features, disc_features * 2, 4, 2, 1),#16 X 16
            self.block(disc_features * 2, disc_features * 4, 4, 2, 1),#8 X 8
            self.block(disc_features * 4, disc_features * 8, 4, 2, 1),# 4 X 4
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(disc_features * 8, 1, kernel_size=4, stride=2, padding=0), # 1 X 1
            nn.Sigmoid(),
        )


    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias = False),
                #Bias is set to false to facilitate Batch normalization
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
    
        )
    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self,noise_channels,img_channels,gen_features):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self.block(noise_channels, gen_features * 16, 4, 1, 0),  # img: 4x4
            self.block(gen_features * 16, gen_features * 8, 4, 2, 1),  # img: 8x8
            self.block(gen_features* 8, gen_features * 4, 4, 2, 1),  # img: 16x16
            self.block(gen_features * 4, gen_features * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                gen_features * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )
    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.gen(x)


#Weight initialization (new Method)
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1)
    print("Discriminator test Passed")
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Generator test Passed")




test()