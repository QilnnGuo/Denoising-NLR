import torch
import torch.nn as nn
import torch.nn.functional as F
    
class UNet_n2n_un(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_n2n_un, self).__init__()

        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=False))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                # m.bias.data.zero_()

    def forward(self, x):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)

        return out

class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included. 
    see our supple for more details. 
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]

        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]

        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

