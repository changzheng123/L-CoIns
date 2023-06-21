# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from numpy import size
from numpy.core.shape_base import block
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from functools import partial
import sys
# import clip

from modeling_finetune import Bert_encoder_onlyLanguage, Block, Block_mae_off, Mlp, _cfg, PatchEmbed, get_sinusoid_encoding_table,Block_crossmodal, MAX_CAP_LEN, Upsample, Biaffine, Conv_Upsample, Conv_Upsample_32, NonLinear,Mlp, GroupingBlock, MixerMlp, Block_D, Conv_Upsample_multiscale,Conv_Upsample_8
from dino_vision_transformer import vit_small
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

# __all__ = [
#     'colorization_mae_base_patch16_224', 
#     'colorization_mae_large_patch16_224', 
# ]

class Colorization_VisionTransformerEncoder_off(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_mae_off(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + self.pos_embed[:, 1:, :]

        B, _, C = x.shape
        # x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

###############################################
#                    decoder
################################################

class Colorization_VisionTransformerDecoder_group_post(nn.Module):# 
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,depth_mlp=4,attn_mode = '',upsample = False
                 , if_classifier=False,num_group_token=10):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 2 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.upsample = upsample

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        ########################################
        self.depth = depth
        blocks = []
        for i in range(self.depth):
            blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values))
        self.blocks_poc = nn.ModuleList(blocks)
        
        self.group_block = GroupingBlock(dim=embed_dim,
                    out_dim=embed_dim,
                    num_heads=num_heads,
                    num_group_token=num_group_token,
                    num_output_group=num_group_token,
                    norm_layer=norm_layer,
                    hard=True,
                    gumbel=True
                    )
        self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dim))
        trunc_normal_(self.group_token, std=.02)
        # dircetion 组件
        num_direction = 12
        self.num_direction = num_direction
        self.d_vectors =  torch.tensor([[[math.cos(2 * math.pi / 12 * float(i)), math.sin(2 * math.pi / 12 * float(i))] for i in range(num_direction)]]).cuda() #[B, 12, 2]
        self.p_coord = self.get_patch_coord() # [B 196 2]
        
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.upsample: 
            self.conv_upsample = Conv_Upsample(if_classifier)
        else: 
            self.depth_mlp = depth_mlp
            blocks_mlp = []
            for i in range(self.depth_mlp):
                blocks_mlp.append(Mlp(embed_dim))
            self.blocks_mlp = nn.ModuleList(blocks_mlp)
            # 最后加一层conv
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                            padding=1, bias=False)
        ########################################
        

        self.token_type_embeddings = nn.Embedding(3, embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_patch_coord(self): 
        x = torch.FloatTensor([i * 16 + 8 for i in range(14)])
        y = torch.FloatTensor([i * 16 + 8 for i in range(14)])
        # print(x)
        grid_x, grid_y = torch.meshgrid(x, y)
        # print(grid_x.shape)
        a = torch.cat(tuple(torch.dstack([grid_x, grid_y]))).unsqueeze(0)
        return a.cuda().detach() # [1 , 196 , 2]

    def get_direc_index_hard(self, attn): #
        bs, Ng, Np = attn.shape
        attn_sum = (attn == 1).sum(dim=2)
        # print(attn_sum.shape)
        attn_sum[attn_sum == 0] = 1
        # print(attn_sum[0])
        inst_coord = (attn/attn_sum.unsqueeze(-1)) @ self.p_coord.repeat(bs,1,1) # [bs, Ng, Np] @ [bs, Np, 2] = [bs, Ng, 2]
        inst_relation = inst_coord.unsqueeze(1).repeat(1, Ng, 1, 1) - inst_coord.unsqueeze(2).repeat(1, 1, Ng, 1)
        # print('patch_relation', inst_relation.shape)
        inst_relation = inst_relation.view(bs, -1, 2) # bs Ng*Ng 2
        # print('patch_relation', inst_relation.shape)
        direction_cos = inst_relation @ self.d_vectors.repeat(bs, 1, 1).transpose(1, 2)  # bs n*n 2 @ bs 2 12 = bs n*n 12
        # print('direction_cos', direction_cos.shape)
        d_index = torch.argmax(direction_cos, dim=-1)
        # print('direction_index', direction_index.shape)
        return d_index, inst_coord

    def forward(self, x, obj, col, occm):
        # attn3
        # print(self.depth)
        # print(len(self.blocks_p))
        # print(len(self.blocks_cross))
        # x.shape = (B,N_p,Dim) 
        
        x_type = self.token_type_embeddings(torch.zeros((x.size()[0],x.size()[1])).cuda().long())
        obj_type = self.token_type_embeddings(torch.full_like(obj[:,:,0], 1).cuda().long())

        x = x + x_type
        obj = obj + obj_type

        group_tokens = self.group_token.repeat(x.size()[0],1,1)

        po = torch.cat([x, obj, group_tokens], dim=1)
        # pc = torch.cat([x, col], dim=1)
        for i in range(self.depth):
            po = self.blocks_poc[i](po)
        p = po[:,0:x.shape[1],:]
        o = po[:,x.shape[1]:x.shape[1]+obj.shape[1],:]
        g = po[:,x.shape[1]+obj.shape[1]:,:]

        g, attn_dict = self.group_block(p,g)
        d_index, inst_coord = self.get_direc_index_hard(attn_dict['hard'].squeeze())
        v_features = g 
        l_features = o
        ################ 
        if self.upsample: # deconv 
            # B x N x dim(768) -> B x N x dim(512)
            p = self.head(self.norm(p))
            bs = p.shape[0]
            size = int(math.sqrt(p.shape[1]))
            dim = p.shape[-1]
            # B x dim(512) x N 
            p = p.permute(0,2,1)
            p = p.reshape(bs,dim,size,size)
            p, pred_label = self.conv_upsample(p)
        else:
            # print('p.shape:',p.shape)
            for i in range(self.depth_mlp):# 过mlp
                p = self.blocks_mlp[i](p)
            
            p = self.head(self.norm(p)) # return ab [B, N, 2*16^2]
            p = rearrange(p, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=int(math.sqrt(p.shape[1])), w=int(math.sqrt(p.shape[1])),c=2,p1=int(math.sqrt(p.shape[-1]/2)))
            p = self.conv(p)
        attn_ ={'coord':inst_coord, 'hard': attn_dict['hard'].squeeze(),'soft':attn_dict['soft'].squeeze()} 
        return p, None, v_features, l_features, pred_label, attn_

#################################################
#                    main model
#################################################

class Colorization_VisionTransformer_group_post(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=768, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=True,
                 attn_mode='',
                 upsample = False,
                 if_class = False,
                 if_contrast = False,
                 num_group_token = 20,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        
        self.encoder = Colorization_VisionTransformerEncoder_off(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = Colorization_VisionTransformerDecoder_group_post(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            attn_mode = attn_mode,
            upsample = upsample,
            if_classifier = if_class,
            num_group_token = num_group_token)

        self.depth = encoder_depth + decoder_depth

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        
        # text_encoder
        self.text_encoder = Bert_encoder_onlyLanguage(decoder_embed_dim)

        self.contrast = if_contrast
        if if_contrast:
            self.proj_img = Mlp(decoder_embed_dim)
            self.proj_txt = Mlp(decoder_embed_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, cap):
        
        x_vis = self.encoder(x) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        # print("x_vis.shape",x_vis.shape)

        obj, col, occm = self.text_encoder(cap,x_vis)

        #  the shape of x is [B, N, 2 * 16 * 16]
        x, occm_pred, v_features, l_features, pred_label, attn = self.decoder(x_vis, obj, col, occm) # [B, N, 2* 16 * 16]

        if self.contrast:
            v_features = self.proj_img(v_features)
            l_features = self.proj_txt(l_features)
        else:
            v_features = None
            l_features = None

        return x, occm_pred, v_features, l_features, pred_label, attn

#################################################
#                    register model
#################################################

@register_model
def colorization_vit_base_patch16_224_group_post(pretrained=False, **kwargs):
    model = Colorization_VisionTransformer_group_post(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=512,
        decoder_embed_dim=1024, 
        decoder_depth=12,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        upsample = True,
        if_class = True,
        num_group_token = 15,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

