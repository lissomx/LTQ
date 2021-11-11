import torch
from torch import nn
from torch.nn import functional as F

# The code is mainly from： 
# 1. https://github.com/rosinality/vq-vae-2-pytorch/blob/master/train_vqvae.py
# 2. torchvision.models.resnet 


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean(dim=3)
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


def conv3x3x(in_planes, out_planes, stride=1, groups=1, dilation=1):
    if stride>0:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, inplanes, outplanes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        width = min([inplanes,outplanes,32])//2
        # Both self.conv2 and self.res_connect layers res_connect the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.res_connect = self._get_res_connect(inplanes, outplanes, stride, norm_layer)

    def _get_res_connect(self, inplanes, outplanes, stride, norm_layer):
        if stride > 1:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 4,2,1),
                norm_layer(outplanes),
            )
        if stride < -1:
            return nn.Sequential(
                nn.ConvTranspose2d(inplanes, outplanes, 4,2,1),
                norm_layer(outplanes),
            )
        elif inplanes != outplanes:
            return nn.Sequential(
                conv1x1(inplanes, outplanes, stride),
                norm_layer(outplanes),
            )
        return None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.res_connect is not None:
            identity = self.res_connect(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck1x1(nn.Module):

    expansion = 4
    def __init__(self, inplanes, outplanes=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        outplanes = outplanes or inplanes
        width = min([inplanes,outplanes])//2
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv3 = conv1x1(width, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.res_connect = self._get_res_connect(inplanes, outplanes, norm_layer)

    def _get_res_connect(self, inplanes, outplanes, norm_layer):
        if inplanes != outplanes:
            return nn.Sequential(
                conv1x1(inplanes, outplanes),
                norm_layer(outplanes),
            )
        return None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.res_connect is not None:
            identity = self.res_connect(x)

        out += identity
        out = self.relu(out)

        return out



class Encoder(nn.Module):
    def __init__(self, plan, n_res_block):
        super().__init__()

        blocks = [
            nn.Conv2d(plan[0], plan[1], 4, stride=2, padding=1), 
            nn.BatchNorm2d(plan[1]),
            nn.ReLU(inplace=True),
        ]

        for i,o in zip(plan[1:-1], plan[2:]):
            blocks += [
                Bottleneck(i, o, 2), 
                Bottleneck(o, o, 1),
            ]

        for _ in range(n_res_block):
            blocks.append(ResBlock(plan[-1], plan[-1]//2))

        blocks += [nn.Conv2d(plan[-1], plan[-1], 3, padding=1, bias=False),]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, plan, n_res_block):
        super().__init__()

        blocks = [nn.Conv2d(plan[0], plan[0], 3, padding=1, bias=False)]

        for i in range(n_res_block):
            blocks.append(ResBlock(plan[0], plan[0]//2))

        for i,o in zip(plan[:-2], plan[1:-1]):
            blocks += [
                Bottleneck(i, o, -2), 
                Bottleneck(o, o, 1),
            ]

        blocks += [ nn.ConvTranspose2d(plan[-2], plan[-1], 4, stride=2, padding=1) ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, ae_plan, n_res_block=2, n_embed=256, **_):
        super().__init__()

        self.enc_b = Encoder(ae_plan, n_res_block)
        self.quantize_conv_b = nn.Conv2d(ae_plan[-1], ae_plan[-1], 1)
        self.quantize_b = Quantize(ae_plan[-1], n_embed)
        self.dec = Decoder(ae_plan[::-1],n_res_block)

    def forward(self, input):
        quant_b, diff, id_b = self.encode(input)
        dec = self.decode(quant_b)

        return dec, diff, id_b

    def encode(self, input):
        enc_b = self.enc_b(input)

        b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        return quant_b, diff_b, id_b

    def decode(self, quant_b):
        dec = self.dec(quant_b)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec