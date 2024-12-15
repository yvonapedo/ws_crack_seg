""" Full assembly of the parts to form the complete network """

from einops import rearrange, repeat

from .model_cmt.cmt_module import IRFFN
from .unet_parts import *
from .Transformer import Transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
from models_shiftvit.shiftvit import BasicLayer
from functools import partial
from .hilo import *
from .splat import SplAtConv2d


class Unet_Sft_Hilo_Splat(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=True):
        super(Unet_Sft_Hilo_Splat, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

        in_dim = [128, 256, 512, 1024]

        self.TFC_S1 = TFC(n_div=4, img_size=256, patch_size=8, in_chans=128, num_classes=1, embed_dim=128,
                          depths=(2, 2, 6, 2), mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=None,
                          act_layer=nn.GELU, patch_norm=True)

        self.TFC_S2 = TFC(n_div=4, img_size=512, patch_size=8, in_chans=512, num_classes=1, embed_dim=512,
                          depths=(2, 2, 6, 2), mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=None,
                          act_layer=nn.GELU, patch_norm=True)

        self.TFC_S3 = TFC(n_div=4, img_size=1024, patch_size=8, in_chans=1024, num_classes=1, embed_dim=1024,
                          depths=(2, 2, 6, 2), mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, norm_layer=None,
                          act_layer=nn.GELU, patch_norm=True)

        self.non_linear1 = _make_layer(Bottleneck, in_dim[0], in_dim[0] // 4, 3, stride=1)
        self.non_linear2 = _make_layer(Bottleneck, 512, 512 // 4, 3, stride=1)

        self.non_linear3 = _make_layer(Bottleneck, 1024, 1024 // 4, 3, stride=1)

        self.max_pool1 = nn.AdaptiveMaxPool2d((16, 8))
        self.max_pool2 = nn.AdaptiveMaxPool2d((16, 8))
        self.max_pool3 = nn.AdaptiveMaxPool2d((16, 8))

        self.conv_128 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_512 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_1024 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

        )

        self.conv_1536 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_256 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gf1_2 = GroupFusion(64, 64, start=True)
        self.gf2_3 = GroupFusion(128, 64, start=True)
        self.gf3_4 = GroupFusion(256, 64, start=True)
        self.gf4_5 = GroupFusion(512, 64, start=True)

        self.gf12_23 = GroupFusion(64, 64, start=True)
        self.gf34_45 = GroupFusion(64, 64, start=True)

        self.gfinal = GroupFusion(64, 64, start=True)
        self.conv_768 = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.HiLo = HiLo(512, 512)

    def forward(self, x):
        # print(x.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)

        c2 = x2  # keep #torch.Size([2, 128, 128, 128])
        # tfc
        p2 = self.non_linear1(x2)
        p2 = self.max_pool1(p2)
        y2, y_mid_2 = self.TFC_S1(p2)  # torch.Size([2, 128, 128, 128])
        y2 = self.conv_128(y2)
        # print(y2.size)
        # print("----")
        # y2 = y2[:, :, :64, 448:]
        y2 = y2[:, :, :64, 448:]
        # down
        x3 = self.down2(c2)
        c3 = x3  # keep

        # cat
        # print(x3.size())
        # print(y2.size())

        t3 = torch.cat([x3, y2], dim=1)  # add tfc # torch.Size([2, 512, 64, 64])

        p3 = self.non_linear2(t3)
        p3 = self.max_pool2(p3)
        y3, y_mid_3 = self.TFC_S2(p3)  # torch.Size([2, 512, 16, 8])
        y3 = self.conv_512(y3)

        # down
        x4 = self.down3(c3)
        c4 = x4  # keep
        # cat
        t4 = torch.cat([x4, y3], dim=1)  # add tfc torch.Size([2, 1024, 32, 32])

        p4 = self.non_linear3(t4)
        p4 = self.max_pool3(p4)
        y4, y_mid_4 = self.TFC_S3(p4)  # torch.Size([2, 1024, 96, 64])
        y4 = self.conv_1024(y4)  # torch.Size([2, 1024, 32, 16])
        y4 = y4[:, :, 16:, 16:]  # torch.Size([2, 1024, 16, 16])

        # down
        x5 = self.down4(c4)  # torch.Size([2, 512, 16, 16])
        # cat
        t5 = torch.cat([x5, y4], dim=1)  # add tfc torch.Size([2, 1536, 16, 16])
        t5 = self.conv_1536(t5)
        x5 = torch.mul(x5, t5)
        x5 = self.HiLo(x5)

        q1 = x1  # torch.Size([2, 64, 256, 256])
        q2 = c2  # torch.Size([2, 128, 128, 128])
        q3 = c3  # torch.Size([2, 256, 64, 64])
        q4 = c4  # torch.Size([2, 512, 32, 32])
        q5 = x5  # torch.Size([2, 512, 16, 16])

        # q2= self.conv_256(q2) #torch.Size([2, 64, 256, 256])
        q2 = q2[:, :64, :, :]
        q2 = torch.nn.functional.interpolate(q2, scale_factor=2, mode='nearest')
        f1_2_l, f1_2 = self.gf1_2(q2, q1)

        q3 = q3[:, :128, :, :]
        q3 = torch.nn.functional.interpolate(q3, scale_factor=2, mode='nearest')
        f2_3_l, f2_3 = self.gf2_3(q3, c2)

        q4 = q4[:, :256, :, :]
        q4 = torch.nn.functional.interpolate(q4, scale_factor=2, mode='nearest')
        f3_4_l, f3_4 = self.gf3_4(q4, c3)

        q5 = q5[:, :512, :, :]
        q5 = torch.nn.functional.interpolate(q5, scale_factor=2, mode='nearest')
        f4_5_l, f4_5 = self.gf4_5(q5, c4)

        f2_3 = torch.nn.functional.interpolate(f2_3, scale_factor=2, mode='nearest')
        f4_5 = torch.nn.functional.interpolate(f4_5, scale_factor=2, mode='nearest')

        f12_23_l, f12_23 = self.gf12_23(f1_2, f2_3)
        f34_45_l, f34_45 = self.gf34_45(f3_4, f4_5)

        f34_45 = torch.nn.functional.interpolate(f34_45, scale_factor=2, mode='nearest')
        f12_23 = F.max_pool2d(f12_23, kernel_size=2, stride=2)

        final_l, final = self.gfinal(f34_45, f12_23)
        final = F.max_pool2d(final, kernel_size=2, stride=2)
        final = F.max_pool2d(final, kernel_size=2, stride=2)

        logits = self.outc(final)
        return logits


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


class TFC(nn.Module):
    def __init__(self, n_div, img_size, patch_size, in_chans, num_classes, embed_dim, depths, mlp_ratio, drop_rate,
                 drop_path_rate, norm_layer, act_layer, patch_norm):
        super(TFC, self).__init__()

        ######################
        self.n_div = n_div  # 12
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.patch_norm = patch_norm
        # self.num_features = int(self.embed_dim * 2 ** (- 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            # norm_layer=self.norm_layer if self.patch_norm else None)
            norm_layer=self.norm_layer)

        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=self.drop_rate)
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        self.use_checkpoint = False

        self.layer = BasicLayer(dim=int(self.embed_dim * 2),
                                n_div=self.n_div,
                                input_resolution=(self.patches_resolution[0] // (2),
                                                  self.patches_resolution[1] // (2)),
                                depth=self.depths[0],
                                mlp_ratio=self.mlp_ratio,
                                drop=self.drop_rate,
                                drop_path=dpr[sum(self.depths[:0]):sum(self.depths[:0 + 1])],
                                # norm_layer=self.norm_layer,
                                norm_layer=None,
                                downsample=PatchMerging,
                                use_checkpoint=self.use_checkpoint,
                                act_layer=self.act_layer)

        R = 3.6
        self.irffn = IRFFN(in_chans, R)

    def forward(self, x, mask=None):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        x = self.patch_embed(x)  # torch.Size([2, 96, 4, 2])

        x = self.pos_drop(x)  # torch.Size([2, 96, 4, 2])
        x = self.layer(x)
        x = self.irffn(x)
        x_mid = x
        return x, x_mid


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
            stride=1,  # if self.avd else stride_3x3,
            padding=1,
            dilation=(1, 1), groups=1, bias=True,
            radix=2, reduction_factor=4,
            rectify=False, rectify_avg=False,
            # norm = None,
            dropblock_prob=0.0)

    def forward(self, f_r, f_l):
        # f_r = self.gf1(f_r)  # chs 768
        f_r = self.SplAtConv2d(f_r)  # chs 768
        f12 = self.gf2(torch.cat((f_r, f_l), dim=1))  # chs 768 torch.Size([2, 64, 256, 256])

        return f12, self.up2x(f12)


class UpSampling2x(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, features):
        return self.up_module(features)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int, tuple): Image size.
        patch_size (int, tuple): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, (2, 2), stride=2, bias=False)
        # self.norm = norm_layer(dim)

    def forward(self, x):
        # x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        """ We use GroupNorm (group = 1) to approximate LayerNorm
        for [N, C, H, W] layout"""
        super(GroupNorm, self).__init__(num_groups, num_channels)

