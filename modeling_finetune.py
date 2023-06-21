# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple

from transformers import BertTokenizer, BertConfig
from transformers import BertModel

import sys
from datasets import MAX_CAP_LEN

from einops.einops import rearrange


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class Sobel_conv(nn.Module):
    def __init__(self):
        super(Sobel_conv, self).__init__()
        kernel_v = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
    
    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss

        """
        
        # transpose labels into labels onehot
        label_onehot = labels

        # calculate log
        log_p = torch.nn.functional.log_softmax(logits,dim=1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class Attention_g(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention_g(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x

class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'

class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=True):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixerMlp(Mlp):
    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)#(3, B, num_heads, N, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)#( B, num_heads, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_D(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, d):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) #(3, B, num_heads, N, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)#( B, num_heads, N, N)
        attn = self.attn_drop(attn)
        attn = attn.unsqueeze(-1).repeat(1,1,1,1,v.shape[-1]) #( B, num_heads, N, N, head_dim)

        # d (B, N, N, dim )
        d = d.reshape(B, N, N, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(B, num_heads, N, N, head_dim)
        v = v.unsqueeze(2).repeat(1,1,N,1,1) + d# ( B, num_heads, N, N, head_dim)
        
        x = torch.sum(attn * v, dim=3) # ( B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_crossmodal(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.k_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.v_linear = nn.Linear(dim, all_head_dim , bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, obj, col, occm=None):

        B, N_p, C = x.shape 
        B, N_o, C = obj.shape

        qkv_bias = None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias


        q = F.linear(input=x, weight=self.q_linear.weight, bias=q_bias).reshape(B, N_p, self.num_heads, -1).permute( 0, 2, 1, 3)
        k = F.linear(input=obj, weight=self.k_linear.weight, bias=k_bias).reshape(B, N_o, self.num_heads, -1).permute( 0, 2, 1, 3)
        v = F.linear(input=obj, weight=self.v_linear.weight, bias=v_bias).reshape(B, N_o, self.num_heads, -1).permute( 0, 2, 1, 3)

        q = q * self.scale # [B, num_heads, N_p, dim]
        attn = (q @ k.transpose(-2, -1)) # [B, num_heads, N_p, N_o]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_p, -1)
        # print("bolck_corss:",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn.transpose(1, 2).reshape(B, N_p, -1)#[B, N_p, num_head*N_o]

class Attention_mae_off(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_mae_off(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_mae_off(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block_crossmodal(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_obj = norm_layer(dim)

        self.attn_1 = Attention_crossmodal(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.attn_2 = Attention_crossmodal(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, obj, col=None, occm = None,attn_mask=None):
        if self.gamma_1 is None:
            x_ = self.norm1(x)
            obj_ = self.norm_obj(obj)

            x_, attn_map = self.attn_1(x_,obj_,col,occm) # attn_map.shape = B x num_head x N_p x N_l
            obj_, attn_map = self.attn_2(obj_,x_,col,occm)
            # print('attn_map.shape',attn_map.shape)
            x_ = x + self.drop_path(x_)
            obj_ = obj + self.drop_path(obj_)

            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_))) 
            obj_ = obj_ + self.drop_path(self.mlp(self.norm2(obj_))) 
        else:
            x_ = self.norm1(x)
            obj_ = self.norm_obj(obj)
            
            x_, attn_map = self.attn_1(x_,obj,col,occm) # attn_map.shape = B x N_p x N_l
            obj_, attn_map = self.attn_2(obj_,x_,col,occm)

            x_ = x + self.drop_path(self.gamma_1 * x_)
            obj_ = obj + self.drop_path(self.gamma_1 * obj_)

            x_ = x_ + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_))) 
            obj_ = obj_ + self.drop_path(self.gamma_2 * self.mlp(self.norm2(obj_))) 
        return x_, obj_

class Block_D(nn.Module):

    def __init__(self, dim, d_emb_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_d = norm_layer(dim)
        self.attn = Attention_D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.direc_proj = nn.Linear(d_emb_dim, dim)
        
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, direc_emb):
        if self.gamma_1 is None:
            direc_emb = self.direc_proj(direc_emb)
            x = x + self.drop_path(self.attn(self.norm1(x),self.norm_d(direc_emb)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            direc_emb = self.direc_proj(direc_emb)
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),self.norm_d(direc_emb)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation


    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, torch.tensor(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, torch.tensor(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1) #

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        # biaffine = torch.sigmoid(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'

class Bert_encoder_onlyLanguage(nn.Module):
    def __init__(self,decoder_dim):
        super().__init__()
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        model_name = 'bert-base-uncased'
        model_config = BertConfig.from_pretrained(model_name)
        model_config.output_hidden_states = True
        # model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained(model_name,config = model_config)

        self.mlp_arc_object = NonLinear(
            input_size = 768,
            hidden_size = decoder_dim,
            activation = nn.ReLU())
        

    def forward(self,txts,vis=None):
        token_ids = []
        for txt in txts:
            token_id = self.tokenizer.encode(txt,add_special_tokens=False,max_length=MAX_CAP_LEN, pad_to_max_length=True)
            token_ids.append(token_id)
        token_tensor = torch.LongTensor(token_ids).cuda()
        cap_emb = self.bert_model(token_tensor)['last_hidden_state']
        obj_emb = self.mlp_arc_object(cap_emb) # b x N_l x dim
        col_emb = None
        # print(obj_emb.shape)
        
        arc_logit = None

        return obj_emb, col_emb, arc_logit

def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv3x3_in_relu(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.InstanceNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block

def conv3x3_tanh(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding and tanh"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.Tanh()
    )
    return block

class Conv_Upsample(nn.Module):
    def __init__(self,if_class=True):
        super(Conv_Upsample, self).__init__()

        self.if_class = if_class

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_in_relu(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_in_relu(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_in_relu(64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.r4 = nn.ReLU(True)
        self.c4 = conv3x3_tanh(32, 2)
        
        if if_class:
            self.up3_classify = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
            self.r3_classify = nn.ReLU(True)
            self.c3_classify = conv3x3_in_relu(128, 128)

            self.up4_classify = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
            self.r4_classify = nn.ReLU(True)
            self.c4_classify = conv3x3_in_relu(128, 128)

            self.classifier = nn.Linear(128,313)

    def forward(self, p):
        """
        :param img_l: batch x 1 x ih x iw
        """
        
        output_1 = self.up1(p)
        output_1 = self.r1(output_1)
        output_1 = self.c1(output_1)# 256 x 28 x28
        
        output_2 = self.up2(output_1)
        output_2 = self.r2(output_2)
        output_2 = self.c2(output_2)# 128 x 56 x 56
        
        output_3 = self.up3(output_2)
        output_3 = self.r3(output_3)
        output_3 = self.c3(output_3)# 64 x 112 x 112

        output_4 = self.up4(output_3)# 32 x 224 x 224
        output_4 = self.r4(output_4)
        output = self.c4(output_4)# 2 x 224 x 224
        pred_label = None
        if self.if_class:
            feature = self.up3_classify(output_2)
            feature = self.r3_classify(feature)
            feature = self.c3_classify(feature)# 128 x 112 x 112

            feature = self.up4_classify(feature)# 128 x 224 x 224
            feature = self.r4_classify(feature)
            feature = self.c4_classify(feature)# 128 x 224 x 224

            feature = feature.flatten(-2).transpose(1,2)

            pred_label = self.classifier(feature)

        return output, pred_label

class Conv_Upsample_32(nn.Module):
    def __init__(self,if_class=True):
        super(Conv_Upsample_32, self).__init__()

        self.if_class = if_class

        self.up0 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True)
        self.r0 = nn.ReLU(True)
        self.c0 = conv3x3_in_relu(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_in_relu(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_in_relu(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_in_relu(64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.r4 = nn.ReLU(True)
        self.c4 = conv3x3_tanh(32, 2)
        
        if if_class:
            self.up3_classify = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
            self.r3_classify = nn.ReLU(True)
            self.c3_classify = conv3x3_in_relu(128, 128)

            self.up4_classify = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
            self.r4_classify = nn.ReLU(True)
            self.c4_classify = conv3x3_in_relu(128, 128)

            self.classifier = nn.Linear(128,313)

    def forward(self, p):
        """
        :param img_l: batch x 1 x ih x iw
        """
        
        output_0 = self.up0(p)
        output_0 = self.r0(output_0)
        output_0 = self.c0(output_0)# 256 x 28 x28
        
        output_1 = self.up1(output_0)
        output_1 = self.r1(output_1)
        output_1 = self.c1(output_1)# 256 x 28 x28

        output_2 = self.up2(output_1)
        output_2 = self.r2(output_2)
        output_2 = self.c2(output_2)# 128 x 56 x 56
        
        output_3 = self.up3(output_2)
        output_3 = self.r3(output_3)
        output_3 = self.c3(output_3)# 64 x 112 x 112

        output_4 = self.up4(output_3)# 32 x 224 x 224
        output_4 = self.r4(output_4)
        output = self.c4(output_4)# 2 x 224 x 224
        pred_label = None
        if self.if_class:
            feature = self.up3_classify(output_2)
            feature = self.r3_classify(feature)
            feature = self.c3_classify(feature)# 128 x 112 x 112

            feature = self.up4_classify(feature)# 128 x 224 x 224
            feature = self.r4_classify(feature)
            feature = self.c4_classify(feature)# 128 x 224 x 224

            feature = feature.flatten(-2).transpose(1,2)

            pred_label = self.classifier(feature)

        return output, pred_label


class Conv_Upsample_8(nn.Module):
    def __init__(self,if_class=True):
        super(Conv_Upsample_8, self).__init__()

        self.if_class = if_class

        self.up1 = nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_in_relu(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_in_relu(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_tanh(32, 2)
        
        if if_class:

            self.up3_classify = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)
            self.r3_classify = nn.ReLU(True)
            self.c3_classify = conv3x3_in_relu(128, 128)

            self.classifier = nn.Linear(128,313)

    def forward(self, p):
        """
        :param img_l: batch x 1 x ih x iw
        """
        
        output_1 = self.up1(p)
        output_1 = self.r1(output_1)
        output_1 = self.c1(output_1)# 256 x 28 x28
        
        output_2 = self.up2(output_1)
        output_2 = self.r2(output_2)
        output_2 = self.c2(output_2)# 128 x 56 x 56

        output_3 = self.up3(output_2)# 32 x 224 x 224
        output_3 = self.r3(output_3)
        output = self.c3(output_3)# 2 x 224 x 224
        pred_label = None
        if self.if_class:

            feature = self.up3_classify(output_2)# 128 x 224 x 224
            feature = self.r3_classify(feature)
            feature = self.c3_classify(feature)# 128 x 224 x 224

            feature = feature.flatten(-2).transpose(1,2)

            pred_label = self.classifier(feature)

        return output, pred_label

class Conv_Upsample_multiscale(nn.Module):
    def __init__(self, if_class=True):
        super(Conv_Upsample_multiscale, self).__init__()

        self.if_class = if_class

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_in_relu(512, 256)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_in_relu(256, 128)

        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_in_relu(128, 64)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.r4 = nn.ReLU(True)
        self.c4 = conv3x3_tanh(64, 2)
        
        if if_class:
            self.up3_classify = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.r3_classify = nn.ReLU(True)
            self.c3_classify = conv3x3_in_relu(128, 128)

            self.up4_classify = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.r4_classify = nn.ReLU(True)
            self.c4_classify = conv3x3_in_relu(128, 128)

            self.classifier = nn.Linear(128,313)

    def forward(self, p):
        """
        :param img_l: batch x 1 x ih x iw
        """
        
        output_1 = self.up1(p)
        # output_1 = self.r1(output_1)
        output_1 = self.c1(output_1)# 256 x 28 x28
        
        output_2 = self.up2(output_1)
        # output_2 = self.r2(output_2)
        output_2 = self.c2(output_2)# 128 x 56 x 56
        
        output_3 = self.up3(output_2)
        # output_3 = self.r3(output_3)
        output_3 = self.c3(output_3)# 64 x 112 x 112

        output_4 = self.up4(output_3)# 32 x 224 x 224
        # output_4 = self.r4(output_4)
        output = self.c4(output_4)# 2 x 224 x 224
        pred_label = None
        if self.if_class:
            feature = self.up3_classify(output_2)
            # feature = self.r3_classify(feature)
            feature = self.c3_classify(feature)# 128 x 112 x 112

            feature = self.up4_classify(feature)# 128 x 224 x 224
            # feature = self.r4_classify(feature)
            feature = self.c4_classify(feature)# 128 x 224 x 224

            feature = feature.flatten(-2).transpose(1,2)

            pred_label = self.classifier(feature)

        return [output_1, output_2, output_3, output], pred_label

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

