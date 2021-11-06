'''
'symbol_candidates' --
    It contains all the symbol in text. 
    Changing its text can change the generated tensor symbols.
    It currently contains 64 symbols, so the index grid will contains 64 different symbols.

'AnonymousPro-Regular.ttf' --
    The tensor symbol generation requires the font file.
    The font should cover all the symbol in 'symbol_candidates'.
    For aesthetic reasons, the font should be monospace.

'numbers2images'
    This list contains the generated symbol tensors.
    Other files will (and only) access this list to get the tensorised symbols.
'''

from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from einops import rearrange

symbol_candidates = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ$£"

Symbols = []
img_size = 32
_t_to_image = transforms.ToTensor()

for l in symbol_candidates:    
    im = Image.new(mode="L", size=(img_size, img_size))

    draw = ImageDraw.Draw(im)
    draw.line((0, 0, img_size, 0), fill=128)
    draw.line((0, 0, 0, img_size), fill=128)
    draw.line((img_size-1, img_size-1, img_size-1, 0), fill=128)
    draw.line((img_size-1, img_size-1, 0, img_size-1), fill=128)

    font = ImageFont.truetype("AnonymousPro-Regular.ttf", 24)
    draw.text((11,4), l ,fill=256,font=font)

    Symbols.append(_t_to_image(im))

Symbols = torch.stack(Symbols)

def numbers2images(nt, resize=None, channel=1):
    # nt := [batch, H, W]
    nt = Symbols[nt%Symbols.shape[0]]
    y = rearrange(nt, "b h w c hn wn -> b c (h hn) (w wn)")
    y = F.interpolate(y,[resize,resize])
    y = y.expand(-1, channel, -1, -1) * 2 -1
    return y
