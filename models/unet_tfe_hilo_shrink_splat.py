""" Full assembly of the parts to form the complete network """

from einops import rearrange, repeat

from .model_cmt.cmt_module import IRFFN
from .unet_parts import *
from .Transformer import Transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .hilo import *
from .splat import  SplAtConv2d


class Unet_TFE_Hilo_Splat(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=True):
        super(Unet_TFE_Hilo_Splat, self).__init__()
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

        heads = 16
        mlp_dim = 2048
        TFE_depth = [3, 3, 6]
        dim_head = 64
        in_dim = [128, 256, 512, 1024]

        self.TFE_S1 = TFE(in_channel=in_dim[0], out_channel=in_dim[0], img_size=[16, 8], num_patch=128, p_size=1,
                          emb_dropout=0.1,
                          T_depth=TFE_depth[0],
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFE_S2 = TFE(in_channel=in_dim[2], out_channel=in_dim[2], img_size=[16, 8], num_patch=128, p_size=1,
                          emb_dropout=0.1,
                          T_depth=TFE_depth[1],
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFE_S3 = TFE(in_channel=in_dim[3], out_channel=in_dim[3], img_size=[16, 8], num_patch=128, p_size=1,
                          emb_dropout=0.1,
                          T_depth=TFE_depth[2],
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)

        self.non_linear1 = _make_layer(Bottleneck, in_dim[0], in_dim[0] // 4, 3, stride=1)
        self.non_linear2 = _make_layer(Bottleneck,512, 512 // 4, 3, stride=1)

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
        x1 = self.inc(x)
        x2 = self.down1(x1)

        c2 = x2 #keep #torch.Size([2, 128, 128, 128])
        #TFE
        p2 = self.non_linear1(x2)
        p2 = self.max_pool1(p2)
        y2, y_mid_2 = self.TFE_S1(p2) # torch.Size([2, 128, 128, 128])
        y2 = self.conv_128(y2)
        y2 = y2[:, :, :64, 448:]
        #down
        x3 = self.down2(c2)
        c3 = x3 #keep

        #cat
        t3 = torch.cat([x3, y2], dim=1)#add TFE # torch.Size([2, 512, 64, 64])

        p3 = self.non_linear2(t3)
        p3 = self.max_pool2(p3)
        y3, y_mid_3 = self.TFE_S2(p3)  #torch.Size([2, 512, 16, 8])
        y3 = self.conv_512(y3)

        # down
        x4 = self.down3(c3)
        c4 = x4  # keep
        # cat
        t4 = torch.cat([x4, y3], dim=1)  # add TFE torch.Size([2, 1024, 32, 32])

        p4 = self.non_linear3(t4)
        p4 = self.max_pool3(p4)
        y4, y_mid_4 = self.TFE_S3(p4)  # torch.Size([2, 1024, 96, 64])
        y4 = self.conv_1024(y4) #torch.Size([2, 1024, 32, 16])
        y4 = y4[:, :, 16:, 16:]  # torch.Size([2, 1024, 16, 16])

        # down
        x5 = self.down4(c4) #torch.Size([2, 512, 16, 16])
        # cat
        t5 = torch.cat([x5, y4], dim=1)  # add TFE torch.Size([2, 1536, 16, 16])
        t5 = self.conv_1536(t5)
        x5 = torch.mul(x5, t5)
        x5 = self.HiLo(x5)

        q1 = x1 #torch.Size([2, 64, 256, 256])
        q2 = c2 # torch.Size([2, 128, 128, 128])
        q3 = c3 # torch.Size([2, 256, 64, 64])
        q4 = c4 # torch.Size([2, 512, 32, 32])
        q5 = x5 # torch.Size([2, 512, 16, 16])

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



# x.input
# torch.Size([2, 128, 16, 8])
# x.interpolate
# torch.Size([2, 128, 256, 256])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# x.layer
# torch.Size([2, 128, 32, 32])
# x.input
# torch.Size([2, 512, 16, 8])
# x.interpolate
# torch.Size([2, 512, 256, 256])
# shift_feat.shape
# torch.Size([2, 512, 32, 32])
# shortcut.shape
# torch.Size([2, 512, 32, 32])
# shift_feat.shape
# torch.Size([2, 512, 32, 32])
# shortcut.shape
# torch.Size([2, 512, 32, 32])
# x.layer
# torch.Size([2, 512, 32, 32])
# y3.conv_512
# torch.Size([2, 512, 32, 32])
# y3.final
# torch.Size([2, 512, 32, 32])
# x4.down3
# torch.Size([2, 512, 32, 32])
# x.input
# torch.Size([2, 1024, 16, 8])
# x.interpolate
# torch.Size([2, 1024, 256, 256])
# shift_feat.shape
# torch.Size([2, 1024, 32, 32])
# shortcut.shape
# torch.Size([2, 1024, 32, 32])
# shift_feat.shape
# torch.Size([2, 1024, 32, 32])
# shortcut.shape
# torch.Size([2, 1024, 32, 32])
# x.layer
# torch.Size([2, 1024, 32, 32])
# y4.conv_1024
# torch.Size([2, 512, 32, 32])
# y4.final
# torch.Size([2, 512, 16, 16])
# x5.down4
# torch.Size([2, 512, 16, 16])
# t5.cat
# torch.Size([2, 512, 16, 16])
# x.input
# torch.Size([2, 128, 16, 8])
# x.interpolate
# torch.Size([2, 128, 256, 256])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# x.layer
# torch.Size([2, 128, 32, 32])
# x.input
# torch.Size([2, 512, 16, 8])
# x.interpolate
# torch.Size([2, 512, 256, 256])
# shift_feat.shape
# torch.Size([2, 512, 32, 32])
# shortcut.shape
# torch.Size([2, 512, 32, 32])
# shift_feat.shape
# torch.Size([2, 512, 32, 32])
# shortcut.shape
# torch.Size([2, 512, 32, 32])
# x.layer
# torch.Size([2, 512, 32, 32])
# y3.conv_512
# torch.Size([2, 512, 32, 32])
# y3.final
# torch.Size([2, 512, 32, 32])
# x4.down3
# torch.Size([2, 512, 32, 32])
# x.input
# torch.Size([2, 1024, 16, 8])
# x.interpolate
# torch.Size([2, 1024, 256, 256])
# shift_feat.shape
# torch.Size([2, 1024, 32, 32])
# shortcut.shape
# torch.Size([2, 1024, 32, 32])
# shift_feat.shape
# torch.Size([2, 1024, 32, 32])
# shortcut.shape
# torch.Size([2, 1024, 32, 32])
# x.layer
# torch.Size([2, 1024, 32, 32])
# y4.conv_1024
# torch.Size([2, 512, 32, 32])
# y4.final
# torch.Size([2, 512, 16, 16])
# x5.down4
# torch.Size([2, 512, 16, 16])
# t5.cat
# torch.Size([2, 512, 16, 16])
# x.input
# torch.Size([2, 128, 16, 8])
# x.interpolate
# torch.Size([2, 128, 256, 256])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# shift_feat.shape
# torch.Size([2, 128, 32, 32])
# shortcut.shape
# torch.Size([2, 128, 32, 32])
# x.layer
# torch.Size([2, 128, 32, 32])


# x.input
# torch.Size([2, 128, 16, 8])
# x.rearrange
# torch.Size([2, 128, 128])
# x.patch_to_embedding
# torch.Size([2, 128, 128])
# x.pos_embedding
# torch.Size([2, 129, 128])
# x.dropout
# torch.Size([2, 129, 128])
# x.transformer
# torch.Size([2, 129, 128])
# x.to_latent
# torch.Size([2, 129, 128])
# x.NeA
# torch.Size([2, 128, 16, 8])

# x.input
# torch.Size([2, 3, 256, 256])
# x.forward_features
# torch.Size([2, 3, 256, 256])
# x.patch_embed
# torch.Size([2, 96, 64, 64])
# x.pos_drop
# torch.Size([2, 96, 64, 64])
# x.layer
# torch.Size([2, 192, 32, 32])
# x.layer
# torch.Size([2, 384, 16, 16])
# x.layer
# torch.Size([2, 768, 8, 8])
# x.layer
# torch.Size([2, 768, 8, 8])
# x.outlayer
# torch.Size([2, 768, 8, 8])
# x.norm
# torch.Size([2, 768, 8, 8])