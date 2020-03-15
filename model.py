import torch.nn as nn


def init_gan_architecture(latent_size=100, base_size=64, num_channels=3):
    G = nn.Sequential(
        #(H_in - 1) * stride + kernelSize - 2 * padding

        #100 x 1 x 1 -> 1024 x 4 x 4
        nn.ConvTranspose2d(latent_size, base_size * 16, kernel_size=6, stride=4, padding=1, bias=False),
        nn.BatchNorm2d(base_size * 16),
        nn.LeakyReLU(True),
        nn.Dropout(0.2),

        #1024 x 4 x 4 -> 512 x 16 x 16
        nn.ConvTranspose2d(base_size * 16, base_size * 8, kernel_size=6, stride=4, padding=1, bias=False),
        nn.BatchNorm2d(base_size * 8),
        nn.LeakyReLU(True),
        nn.Dropout(0.2),

        #512 x 16 x 16 -> 256 x 64 x 64
        nn.ConvTranspose2d(base_size * 8, base_size * 4, kernel_size=6, stride=4, padding=1, bias=False),
        nn.BatchNorm2d(base_size * 4),
        nn.LeakyReLU(True),
        nn.Dropout(0.2),

        # 256 x 64 x 64 -> 3 x 256 x 256
        nn.ConvTranspose2d(base_size * 4, num_channels, kernel_size=6, stride=4, padding=1, bias=False),
        nn.Tanh()
    )


    D = nn.Sequential(
        # H = (H1 - 1)*stride + HF - 2*padding

        # 3 x 256 x 256 -> 256 x 64 x 64
        nn.Conv2d(num_channels, base_size * 4, 6, 4, 1, bias=False),
        nn.BatchNorm2d(base_size*4),
        nn.LeakyReLU(True),
        nn.Dropout(),

        #256 x 64 x 64 - > 512 x 16 x 16
        nn.Conv2d(base_size * 4, base_size * 8, 6, 4, 1, bias=False),
        nn.BatchNorm2d(base_size*8),
        nn.LeakyReLU(True),
        nn.Dropout(),

        #512 x 16 x 16 - > 1024 x 4 x 4
        nn.Conv2d(base_size * 8, base_size * 16, 6, 4, 1, bias=False),
        nn.BatchNorm2d(base_size*16),
        nn.LeakyReLU(True),
        nn.Dropout(),

        #1024 x 4 x 4 -> 1 x 1 x 1
        nn.Conv2d(base_size*16, 1, 6, 4, 1, bias=False),
        nn.Sigmoid()
    )
    return G, D, latent_size


if __name__ == '__main__':
    G, D = init_gan_architecture()

    print(D.parameters)
    print(G.parameters)
    
    