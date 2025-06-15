from turtle import forward
from numpy import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import torchvision.models as models
from lib.modules import *

import math
from pdb import set_trace as stx
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from lib.PCPart import PhaseCongruencyLayer
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.GateV=nn.Conv2d(dim, dim*2, kernel_size=3,
                                stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        #stx()
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        v1,v2 = self.GateV(v).chunk(2, dim=1)
        v = F.gelu(v1) * v2
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        #import pdb;pdb.set_trace()
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransDownBlock(nn.Module):
    def __init__(self,indim,outdim,num_heads,ffn_expansion_factor,bias,LayerNorm_type):
        super(TransDownBlock,self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.norm1=LayerNorm(indim,LayerNorm_type)
        self.preconv = nn.Sequential(nn.Conv2d(indim,indim//2,kernel_size=3,padding=1,bias=False),
                                    nn.BatchNorm2d(indim//2,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(indim//2,indim//4,kernel_size=3,dilation=2,padding=2,bias=False),
                                    nn.BatchNorm2d(indim//4,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(indim//4,indim//8,kernel_size=3,dilation=4,padding=4,bias=False),
                                    nn.BatchNorm2d(indim//8,affine=False),
                                    nn.ReLU()
                                    )
        self.dlConv1=nn.Conv2d(indim//8, indim//8, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
        self.attn=Attention(indim//8,num_heads,bias)
        self.norm2=LayerNorm(indim//8,LayerNorm_type)
        self.BN=nn.BatchNorm2d(indim//8, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)

        self.dlConv3=nn.Conv2d(indim//8, outdim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        self.BN3=nn.BatchNorm2d(outdim, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.Relu=nn.ReLU()
        self.pool1 = nn.AvgPool2d(3,stride=1,padding=1)
    def forward(self, x):
        x=self.dropout(x)
        x=self.preconv(x)
        x=self.Relu(self.dlConv1(x))
        x=x+self.attn(self.norm2(x))
       # x = x.exp().clamp(max=1e4)
        #x=x/(self.pool1(x)+1e-5)
        
        #x=self.BN(x)
        #x=x+self.attn(self.norm2(x))
        x=self.BN(x)
        
        #x = self.Relu(self.BN2(self.dlConv2(x)))
        #x = x.exp().clamp(max=1e4)
        #x=x/(self.pool1(x)+1e-5)
        x=self.Relu(self.BN3(self.dlConv3(x)))
        #x = x.exp().clamp(max=1e4)
        #x=x/(self.pool1(x)+1e-5)
        #x = F.softmax(x,dim=1)[:,0].unsqueeze(1)
        return x
        

class TransformerBlockDialation(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,dialtion=4,res=True):
        super(TransformerBlockDialation, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.BN=nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.res=res
        self.dlConv=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(dialtion, dialtion), dilation=(dialtion, dialtion))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        #x = x + self.ffn(self.norm2(x))
        if self.res:
            xtmp=self.dlConv(self.norm2(x))
           
            x=x+xtmp
            x=self.BN(x)
        else:
            x =  self.dlConv(self.norm2(x))
            x=self.BN(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x




class CrossCheck(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(CrossCheck, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.qkv3 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv4 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
    def forward(self,x,y):
        b,c,h,w=x.shape
        qkv_x=self.qkv2(self.qkv1(x))
        qkv_y=self.qkv4(self.qkv3(y))
        q_x,k_x,v_x=qkv_x.chunk(3,dim=1)
        q_y,k_y,v_y=qkv_y.chunk(3,dim=1)
        q_xx = rearrange(q_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_xx = rearrange(k_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_xx = rearrange(v_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q_yy = rearrange(q_y, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k_yy = rearrange(k_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v_yy = rearrange(v_x, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q_xx = torch.nn.functional.normalize(q_xx, dim=-1)
        k_xx = torch.nn.functional.normalize(k_xx, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn_x = (q_xx @ k_xx.transpose(-2, -1)) * self.scale
        attn_x = attn_x.softmax(dim=-1)

        outx = (attn_x @ v_xx)

        outx = rearrange(outx, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        outx = self.proj(outx)

        q_yy = torch.nn.functional.normalize(q_yy, dim=-1)
        k_yy = torch.nn.functional.normalize(k_yy, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn_y = (q_yy @ k_yy.transpose(-2, -1)) * self.scale
        attn_y = attn_y.softmax(dim=-1)

        outy = (attn_y @ v_yy)

        outy = rearrange(outy, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        outy = self.proj2(outy)
        return outx,outy 

class SAGNetV2(nn.Module):
    def __init__(self,inp_channels=3,dim=128,bias=False,LayerNorm_type='WithBias'):
        super(SAGNetV2,self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels,dim)
        self.SA1=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=4,res=False)
        self.SA2=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=8,res=False)
        self.SA3=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=16,res=False)
        self.SA4=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=4,res=False)        
        self.SA5=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=8,res=False)
        self.SA6=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=16,res=False)
    def forward(self,x):
        x=self.patch_embed(x)
        x=self.SA1(x)
        x=self.SA2(x)
        x=self.SA3(x)
        x=self.SA4(x)
        x=self.SA5(x)
        x=self.SA6(x)
        
        return x


class SAFeatureExtractor(nn.Module):
    def __init__(self,inp_channels=3,dim=128,bias=False,LayerNorm_type='WithBias'):
        super(SAFeatureExtractor,self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels,dim)
        self.SA1=TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.SA2=TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.SA3=TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)
        self.SA4=TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type)       

    def forward(self,x):
        x=self.patch_embed(x)
        x=self.SA1(x)
        x=self.SA2(x)
        x=self.SA3(x)
        x=self.SA4(x)

        
        return x

class TransformerCrossBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerCrossBlock, self).__init__()

        self.norm1x = LayerNorm(dim, LayerNorm_type)
        self.norm1y = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossCheck(dim, num_heads, bias)
        self.norm2x = LayerNorm(dim, LayerNorm_type)
        self.norm2y = LayerNorm(dim, LayerNorm_type)
        self.ffnx = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffny = FeedForward(dim, ffn_expansion_factor, bias)
        self.dlConv=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.BN=nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    def forward(self, x,y):
        x=self.norm1x(x)
        y=self.norm1y(y)
        x1,y1=self.attn(x,y)
        x=x+1.5*x1
        y=y+1.5*y1
        x=self.BN(x+self.dlConv(x))
        y=self.BN(y+self.dlConv(y))


        return x,y

class SADeepExtract(nn.Module):
    def __init__(self,inp_channels=3,dim=128,bias=False,LayerNorm_type='WithBias'):
        super(SADeepExtract,self).__init__()

        self.SA1=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=4,res=True)
        self.SA2=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=8,res=False)
        self.SA3=TransformerBlockDialation(dim=dim, num_heads=8, ffn_expansion_factor=2,
                                                               bias=bias, LayerNorm_type=LayerNorm_type,dialtion=16,res=True)
     

    def forward(self,x):

        x=self.SA1(x)
        x=self.SA2(x)
        x=self.SA3(x)
       
        return x
class RepAndRel(nn.Module):
    def __init__(self):
        super(RepAndRel,self).__init__()
        self.ScoreMapEstimation=TransDownBlock(indim=128,outdim=1, num_heads=8, ffn_expansion_factor=2,
                                                               bias=False,LayerNorm_type='WithBias')
        
    def forward(self,x):
        score=self.ScoreMapEstimation(x)
        #score=F.softplus(score)
        return score
class RepAndRel2(nn.Module):
    def __init__(self):
        super(RepAndRel2,self).__init__()
        self.ScoreMapEstimation=TransDownBlock(indim=128,outdim=1, num_heads=8, ffn_expansion_factor=2,
                                                               bias=False,LayerNorm_type='WithBias')
        self.PCKP=kpDet()

    def p_x(self,x):
        x=self.PCKP(x)
        p_x=F.softplus(x)
        p_x=p_x/(1+p_x)
        return p_x    
    def forward(self,x,img):
        p_c=self.ScoreMapEstimation(x)
        p_x=self.p_x(img)
        p_y=(p_x*p_c)
        #score=F.softplus(score)
        return p_y

class kpDet(nn.Module):
    def __init__(self):
        super(kpDet,self).__init__()
        self.DownSamp=nn.Conv2d(3,1,3,padding=1,bias=False)
        self.PCPart=PhaseCongruencyLayer()
    def forward(self,x):
        x=self.DownSamp(x)
        x=self.PCPart(x)
        return x
class ConditionalEstimator(nn.Module):
    def __init__(self) -> None:
        super(ConditionalEstimator,self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.preconv = nn.Sequential(nn.Conv2d(128,64,kernel_size=3,padding=1,bias=False),
                                    nn.BatchNorm2d(64,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(64,32,kernel_size=3,dilation=2,padding=2,bias=False),
                                    nn.BatchNorm2d(32,affine=False),
                                    nn.ReLU(),
                                    nn.Conv2d(32,16,kernel_size=3,dilation=4,padding=4,bias=False),
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU()
                                    )
        self.bn1 = nn.Sequential(nn.BatchNorm2d(16,affine=False),
                                nn.ReLU(),
                                nn.InstanceNorm2d(16,affine=False),
                                nn.ReLU())
        self.bn2 = nn.Sequential(nn.BatchNorm2d(16,affine=False),
                                nn.ReLU(),
                                nn.InstanceNorm2d(16,affine=False),
                                nn.ReLU())
        self.bn3 = nn.Sequential(nn.BatchNorm2d(16,affine=False),
                                nn.ReLU(),
                                nn.InstanceNorm2d(16,affine=False),
                                nn.ReLU())
        self.pool1 = nn.AvgPool2d(3,stride=1,padding=1)
        self.pool2 = nn.AvgPool2d(3,stride=1,padding=1)
        self.pool3 = nn.AvgPool2d(3,stride=1,padding=1)
        self.layer1 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU(),
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),
                                    nn.BatchNorm2d(16,affine=False),
                                    nn.ReLU(),
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(16,16,3,padding=1,bias=False),
                            nn.BatchNorm2d(16,affine=False),
                            nn.ReLU())
        self.postconv = nn.Sequential(nn.Conv2d(16,2,3,padding=1,bias=False))


    def LN(self,x,conv,pool,bn):
        x = conv(x)
        x = x.exp().clamp(max=1e4)
        x = x/(pool(x)+1e-5)
        x = bn(x)
        return x
    def forward(self,x):
        x = self.dropout(x)
        x = self.preconv(x)
        x = self.LN(x,self.layer1,self.pool1,self.bn1)
        x = self.LN(x,self.layer2,self.pool2,self.bn2)
        x = self.LN(x,self.layer3,self.pool3,self.bn3)
        x = self.postconv(x)
        x = F.softmax(x,dim=1)[:,0].unsqueeze(1)
        return x
class Superhead(nn.Module):
    def __init__(self) -> None:
        super(Superhead,self).__init__()
        self.PriorEstimator = nn.Sequential(nn.Conv2d(128,1,kernel_size=1))
        
        self.ConditionalEstimator = ConditionalEstimator()
    def p_x(self,x):
        x = self.PriorEstimator(x)
        p_x = F.softplus(x)
        p_x = p_x/(1+p_x)
        #p_x = x/(1+x)
        return p_x
    
    def forward(self,x):
        p_c = self.ConditionalEstimator(x)
        p_x = self.p_x(x)
        # p_y = F.softplus(x1+x2)
        # p_y = p_y/(1+p_y)
        # p_x = self.p_x(x)
        p_y = (p_x*p_c)
        return p_y 
class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()
        
        #self.PC=PhaseCongruencyLayer()
        self.AT1=SAFeatureExtractor()
        self.AT2=SAFeatureExtractor()
        #self.CA1=TransformerCrossBlock(dim=128, num_heads=8, ffn_expansion_factor=2,
        #                                                       bias=False, LayerNorm_type='Withbias')
        self.enc=SADeepExtract()
        #self.det=RepAndRel()
        self.det=Superhead()
    def forward1(self,imgs):
        feat_in = self.AT1(imgs)
        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat,dim=1), score

    def forward2(self,imgs):
        feat_in = self.AT2(imgs)
        feat = self.enc(feat_in)
        score = self.det(feat.pow(2))
        return F.normalize(feat,dim=1), score
            
    def forward(self,img1,img2):
        # import pdb;pdb.set_trace()
        # img1=self.PC(img1)
        # img2=self.PC(img2)
        feat1=self.AT1(img1)
        feat2=self.AT2(img2)
        feat1=self.enc(feat1)
        feat2=self.enc(feat2)
        #feat1,feat2=self.CA1(feat1,feat2)
        #stx()
        score1 = self.det(feat1.pow(2).detach())
        score2 = self.det(feat2.pow(2).detach())
        return {
            'feat': [F.normalize(feat1,dim=1), F.normalize(feat2,dim=1)],
            'score': [score1, score2]
        } 