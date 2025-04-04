from einops import rearrange, repeat
from .model_cmt.cmt_module import IRFFN
from .unet_parts import *
from .Transformer import Transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .hilo import *
from .splat import  SplAtConv2d

class TFE(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super(TFE, self).__init__()

        height, width = img_size

        self.p_size = p_size

        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.NeA = Bottleneck(out_channel, out_channel//4)

    def forward(self, x, mask=None):

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        x = self.patch_to_embedding(x)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_mid = x[:, 0]
        x_mid = self.to_latent(x_mid)
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x = self.NeA(x)

        return x, x_mid


class Bottleneck(nn.Module):
    """Bottleneck block with residual connection"""
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, start=False):  # 768, 384
        super(GroupFusion, self).__init__()
        temp_chs = in_chs
        if start:
            in_chs = in_chs
        else:
            in_chs *= 2

        self.gf1 = nn.Sequential(nn.Conv2d(in_chs, temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))

        self.gf2 = nn.Sequential(nn.Conv2d((temp_chs + temp_chs), temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))
        self.up2x = UpSampling2x(temp_chs, out_chs)

        self.SplAtConv2d = SplAtConv2d(
            in_chs, in_chs, kernel_size=3,
            stride=1, #if self.avd else stride_3x3,
            padding=1,
            dilation = (1, 1), groups = 1, bias = True,
            radix = 2, reduction_factor = 4,
            rectify = False, rectify_avg = False,
            # norm = None,
            dropblock_prob = 0.0)

    def forward(self, f_r, f_l):
        # f_r = self.gf1(f_r)  # chs 768
        f_r = self.SplAtConv2d(f_r)  # chs 768
        f12 = self.gf2(torch.cat((f_r, f_l), dim=1))  # chs 768 torch.Size([2, 64, 256, 256])

        return f12, self.up2x(f12)

class UpSampling2x(nn.Module):
    """Pixel shuffle upsampling"""

    def __init__(self, in_chs: int, out_chs: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_chs, out_chs * 4, 1, bias=False),
            nn.BatchNorm2d(out_chs * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def _make_layer(block, inplanes, planes, blocks, stride=2):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def build_deconv_block(in_ch: int, out_ch: int,kernel: int, stride: int,repeat: int = 1) -> nn.Sequential:

    layers = []
    for _ in range(repeat):
        layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel, stride),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
        in_ch = out_ch  # Update input channels for subsequent blocks
        return nn.Sequential(*layers)

FUSION_CONFIG = [
            # Stage-wise feature integration
            (64, 64),  # gf1_2: Integrate early features
            (128, 64),  # gf2_3: Cross-scale fusion 1
            (256, 64),  # gf3_4: Cross-scale fusion 2
            (512, 64),  # gf4_5: High-level feature integration

            # Multi-resolution fusion
            (64, 64),  # gf12_23: Combine initial fusion outputs
            (64, 64),  # gf34_45: Combine mid-level fusions

            # Final aggregation
            (64, 64)  # gfinal: Final feature synthesis
        ]
class Unet_TFE_Hilo_Splat(nn.Module):
    """Complete U-Net with Transformer Feature Enhancement"""

    def __init__(self, n_channels: int, num_classes: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.inc_1 = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1))

        # Transformer Feature Enhancement
        tfe_config = {
            'img_size': (16, 8),
            'num_patch': 128,
            'p_size': 1,
            'emb_dropout': 0.1,
            'heads': 16,
            'dim_head': 64,
            'mlp_dim': 2048,
            'dropout': 0.1
        }
        self.tfe_stages = nn.ModuleList([
            TFE(128, 128, T_depth=3, **tfe_config),
            TFE(512, 512, T_depth=3, **tfe_config),
            TFE(1024, 1024, T_depth=6, **tfe_config)
        ])

        # Decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

        # Feature refinement
        self.non_linear = nn.ModuleList([
            _make_layer(Bottleneck,128, 32, 3, stride=1),
            _make_layer(Bottleneck,512, 128, 3,stride=1),
            _make_layer(Bottleneck,1024, 256, 3, stride=1)
        ])

        self.max_pool = nn.AdaptiveMaxPool2d((16, 8))
        self.HiLo = HiLo(512, 512)
        self.channel_reducers = nn.ModuleDict({
            '64': nn.Conv2d(128, 64, 1),
            '128': nn.Conv2d(256, 128, 1),
            '256': nn.Conv2d(512, 256, 1),
            '512': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        })
        # Channel reduction layers with systematic naming
        self.q_reducers = nn.ModuleDict({
            '64': nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            '128': nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            '256': nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            '512': nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        })

        self.fusion_layers = nn.ModuleList()
        for in_ch, out_ch in FUSION_CONFIG:
            self.fusion_layers.append(
                GroupFusion(in_chs=in_ch,
                            out_chs=out_ch,
                            start=True)
            )
        # Named accessors for clarity
        self.gf1_2 = self.fusion_layers[0]
        self.gf2_3 = self.fusion_layers[1]
        self.gf3_4 = self.fusion_layers[2]
        self.gf4_5 = self.fusion_layers[3]
        self.gf12_23 = self.fusion_layers[4]
        self.gf34_45 = self.fusion_layers[5]
        self.gfinal = self.fusion_layers[6]

        # Feature upsampling paths
        self.conv_128_256 = build_deconv_block(
            in_ch=128, out_ch=128, kernel=4, stride=4, repeat=2
        ).append(build_deconv_block(128, 256, 1, 1))

        self.conv_512 = build_deconv_block(
            512, 512, 1, 1, repeat=2
        )

        self.conv_1024 = nn.Sequential(
            build_deconv_block(1024, 1024, 1, 1),
            build_deconv_block(1024, 512, 1, 1)
        )

        # Specialized connection modules
        self.conv_1024_512 = build_deconv_block(1024, 512, 1, 1)

    def _make_bottleneck(self, in_chs: int, planes: int, blocks: int) -> nn.Sequential:
        layers = []
        layers.append(Bottleneck(in_chs, planes))
        for _ in range(1, blocks):
            layers.append(Bottleneck(planes * Bottleneck.expansion, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder

        if x.shape[1]==1:
            x1 = self.inc_1(x)
        else:
            x1 = self.inc(x)
        # x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # TFE Processing
        p2 = self.non_linear[0](x2)
        p2 = self.max_pool(p2)
        y2, _ = self.tfe_stages[0](p2)
        y2 = self.conv_128_256(y2)

        y2 = F.interpolate(y2, size=x3.shape[2:], mode='bilinear', align_corners=False)

        t3 = torch.cat([x3, y2], dim=1)
        p3 = self.non_linear[1](t3)
        p3 = self.max_pool(p3)
        y3, _ = self.tfe_stages[1](p3)
        y3 = self.conv_512(y3)
        y3 = F.interpolate(y3, size=x4.shape[2:], mode='bilinear', align_corners=False)

        t4 = torch.cat([x4, y3], dim=1)
        p4 = self.non_linear[2](t4)
        p4 = self.max_pool(p4)
        y4, _ = self.tfe_stages[2](p4)
        y4 = self.conv_1024(y4) #torch.Size([2, 1024, 32, 16])

        y4 = F.interpolate(y4, size=x5.shape[2:], mode='bilinear', align_corners=False)

        # Decoder
        # Feature integration and enhancement
        t5 = torch.cat([x5, y4], dim=1)  # Channel concatenation
        t5 = self.conv_1024_512(t5)  # Channel reduction: 1536 -> 512
        x5 = self.HiLo(x5 * t5)  # Feature modulation and HiLo attention

        # Multi-scale feature extraction
        features = {
            'q1': x1,
            'q2': self.channel_reducers['64'](x2),
            'q3': self.channel_reducers['128'](x3),
            'q4': self.channel_reducers['256'](x4),
            'q5': x5
        }

        # Feature upsampling and fusion
        fused_features = []
        for key in ['q2', 'q3', 'q4', 'q5']:
            features[key] = F.interpolate(features[key], scale_factor=2, mode='nearest')

        # Hierarchical feature grouping
        f1_2_l, f1_2 = self.gf1_2(features['q2'], features['q1'])
        f2_3_l, f2_3 = self.gf2_3(features['q3'], x2)
        f3_4_l, f3_4 = self.gf3_4(features['q4'], x3)
        f4_5_l, f4_5 = self.gf4_5(features['q5'], x4)

        # Intermediate feature processing
        f2_3 = F.interpolate(f2_3, scale_factor=2, mode='nearest')
        f4_5 = F.interpolate(f4_5, scale_factor=2, mode='nearest')

        # Multi-stage fusion
        f12_23_l, f12_23 = self.gf12_23(f1_2, f2_3)
        f34_45_l, f34_45 = self.gf34_45(f3_4, f4_5)

        # Final feature aggregation
        f34_45 = F.interpolate(f34_45, scale_factor=2, mode='nearest')
        f12_23 = F.max_pool2d(f12_23, kernel_size=2, stride=2)

        # Output preparation
        final_l, final = self.gfinal(f34_45, f12_23)
        for _ in range(2):  # Apply twice: 16x16 -> 8x8 -> 4x4
            final = F.max_pool2d(final, kernel_size=2, stride=2)
        return self.outc(final)
