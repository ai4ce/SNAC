import torch 
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li

class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""
    def __init__(self, input_size, hidden_size,device):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size,num_layers=1,batch_first=True)
    def forward(self, x, hidden_state):
        _, h_n = self.rnn(x,hidden_state)
        return h_n
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        return h

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""
    def __init__(self, hidden_size, output_size, device):
        super().__init__()
        self.output_size = output_size
        self.device = device
        self.rec_dec1 = nn.GRUCell(output_size, hidden_size)
        self.dense_dec1 = get_and_init_FC_layer(hidden_size, output_size)
    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)
        # Squeezing
        h_i = h_0.squeeze()
        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)
            x = torch.cat([x, x_i], axis=1)

        return x.view(-1, seq_len, self.output_size)

class RecurrentAE(nn.Module):
    """Recurrent autoencoder
       input: a sequence of obs with size (B,L,51) and hidden state
       output: a sequence of obs with size (B,L,49*3+2)
    """
    def __init__(self,input_size, output_size, hidden_size, device, train=True):
        super().__init__()
        # Encoder and decoder configuration
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.input_size, self.hidden_size, self.device).to(device)
        self.decoder = RecurrentDecoder(self.hidden_size, self.output_size, self.device).to(device)
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def forward(self, x, hidden_state):
        seq_len = x.shape[1]
        h_n = self.encoder(x,hidden_state)
        out = self.decoder(h_n, seq_len)
        return torch.flip(out, [1])

class SNAC_Lnet(nn.Module):
    """Recurrent Lnet
       input: a sequence of obs with size (B,L,51) and hidden state
       output: a sequence of obs with size (B,L,49*3+2)
    """
    def __init__(self,input_size, hidden_size, device, Loss_type="L2"):
        super().__init__()
        # Encoder and decoder configuration
        self.hidden_size = hidden_size
        self.input_size = input_size
        if Loss_type == "L2": 
            self.output_size = 2
        else:
            self.output_size = 26*26 
        self.device = device
        self.Loss_type = Loss_type
        # Encoder and decoder
        self.rnn = nn.LSTM(self.input_size, self.hidden_size,batch_first=True).to(device)
        if Loss_type == "L2":
            self.MLP = nn.Sequential(
                        get_and_init_FC_layer(self.hidden_size,64),
                        nn.ReLU(),
                        get_and_init_FC_layer(64,16),
                        nn.ReLU(),
                        get_and_init_FC_layer(16,self.output_size),
                        nn.ReLU())
        else: 
            self.MLP = nn.Sequential(
                        get_and_init_FC_layer(self.hidden_size,256),
                        nn.ReLU(),
                        get_and_init_FC_layer(256,512),
                        nn.ReLU(),
                        get_and_init_FC_layer(512,self.output_size),
                        nn.LogSoftmax(dim=2))

    def forward(self, x, pos, hidden_state, cell_state):
        """
        x: size (B,L,K), K = 51+51+1 (two obs + action)
        pos: size (B,L,2)
        """
        seq_len = x.shape[1]
        B_size = x.shape[0]
        predicted_pos = []
        input_pos = pos[:,0:1,:]
        for i in range(0, seq_len):
            output, (hidden_state,cell_state) = self.rnn(torch.cat((x[:,i:i+1,:],input_pos),dim=2),(hidden_state,cell_state)) ### output shape = (B,1,hidden size)
            if self.Loss_type == "L2":
                next_pos = self.MLP(output) ## next pos size (B,1,2)
                predicted_pos.append(next_pos)
                input_pos = next_pos
            else:
                next_pos = self.MLP(output) ## next pos size (B,1,576), 26**2
                next_pos = next_pos.view(B_size,1,26,26)
                predicted_pos.append(next_pos)
                if i >= (seq_len-1):
                    break
                input_pos = pos[:,i+1:i+2,:]
        return predicted_pos, hidden_state, cell_state
        
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        c = torch.zeros(1,bsize,self.hidden_size).float().to(self.device)
        return h,c

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:-1]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Residual connection + pos encoding
        # return x + self.pos_encoding[:x.size(0), :]
        return x + self.pe[:x.size(0)]

class AttentionAE(nn.Module):
    def __init__(self, in_features, out_features, max_len, nhead, num_layers):
    #def __init__(self, in_features, out_features, nhead, num_layers):
        super().__init__()
        self.pos = PositionalEncoding(in_features, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead)
        self.attention = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.mlp = nn.Linear(in_features, out_features)

    def forward(self, src):
        encoded = self.pos(src)
        attention = self.attention(encoded)
        #attention = self.attention(src)
        output = self.mlp(attention)
        return attention, output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)     
        x = self.drop(x)
        return x


class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True, grid_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)       
        self.k = nn.Linear(dim, dim, bias=qkv_bias)    
        self.v = nn.Linear(dim, dim, bias=qkv_bias)       
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength
        self.gating_param = nn.Parameter(1*torch.ones(self.num_heads))
        self.apply(self._init_weights)
        if use_local_init:
            self.local_init(locality_strength=locality_strength)
        self.current_grid_size = grid_size
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def get_attention(self, x):
        B, N, C = x.shape  

        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)            

        pos_score = self.pos_proj(self.rel_indices).expand(B, -1, -1,-1).permute(0,3,1,2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn = attn / attn.sum(dim=-1).unsqueeze(-1) 
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
    
    def local_init(self, locality_strength=1.):
        
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1 #max(1,1/locality_strength**.5)
        
        kernel_size = int(self.num_heads**.5)
        center = (kernel_size-1)/2 if kernel_size%2==0 else kernel_size//2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1+kernel_size*h2
                self.pos_proj.weight.data[position,2] = -1
                self.pos_proj.weight.data[position,1] = 2*(h1-center)*locality_distance
                self.pos_proj.weight.data[position,0] = 2*(h2-center)*locality_distance
        self.pos_proj.weight.data *= locality_strength

    def get_rel_indices(self, ):
        H, W = self.current_grid_size
        N = H*W
        rel_indices = torch.zeros(1, N, N, 3)
        indx = torch.arange(W).view(1,-1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(H, H)
        indy = torch.arange(H).view(1,-1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat_interleave(W, dim=0).repeat_interleave(W, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.v.weight.device
        self.rel_indices = rel_indices.to(device)
        
    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.size(1)!=N:
            self.get_rel_indices()

        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., grid_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        self.current_grid_size = grid_size
        
    def _init_weights(self, m):        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_attention_map(self, x, return_map = False):
        self.get_rel_indices()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0) # average over batch
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map))
        dist /= distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist
        
    def get_rel_indices(self, ):
        H, W = self.current_grid_size
        N = H*W
        rel_indices = torch.zeros(1, N, N, 3)
        indx = torch.arange(W).view(1,-1) - torch.arange(W).view(-1, 1)
        indx = indx.repeat(H, H)
        indy = torch.arange(H).view(1,-1) - torch.arange(H).view(-1, 1)
        indy = indy.repeat_interleave(W, dim=0).repeat_interleave(W, dim=1)
        indd = indx**2 + indy**2
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        device = self.qkv.weight.device
        self.rel_indices = rel_indices.to(device)                

            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, grid_size):
        self.attn.current_grid_size = grid_size
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, from timm
    """
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1,patch_size), stride=patch_size)      
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)      
        self.apply(self._init_weights)
        
    def forward(self, x):
        x = self.proj(x)

        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, avrg_img_size=147, patch_size=7, in_chans=1, embed_dim=64, depth=8,
                 num_heads=9, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 gpsa_interval=[-1, -1], locality_strength=1., use_pos_embed=True):
        
        super().__init__()
        self.depth = depth
        embed_dim *= num_heads
        self.num_features = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed
        self.avrg_img_size = avrg_img_size

        if isinstance(self.avrg_img_size, int):
            # img_size = (7, 21) # for implicit
            img_size = (1,105) # for explicit

        if isinstance(patch_size, int):
            # self.patch_size = (1,7) for implicit
            self.patch_size = (1,1) # for explicit

        self.in_chans = in_chans
        
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim,
                    img_size[0] // self.patch_size[0],
                    img_size[1] // self.patch_size[1])
                )
            
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
                
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i>=gpsa_interval[0]-1 and i<gpsa_interval[1] else            
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False,)
            for i in range(depth)])
                
        self.norm = norm_layer(embed_dim)

        # head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        #self.head = nn.Linear(self.num_features, in_chans*self.patch_size[0]*self.patch_size[1]) #implicit
        
        #explicit
        self.head = nn.Linear(9, 1)
        self.mlp = nn.Linear(105, 2)
        
    def seq2img(self, x, img_size):
        """
        Transforms sequence back into image space, input dims: [batch_size, num_patches, channels]
        output dims: [batch_size, channels, H, W]
        """
        x = x.view(x.shape[0], x.shape[1], self.in_chans, self.patch_size[0], self.patch_size[1])
        x = x.chunk(x.shape[1], dim=1)
        x = torch.cat(x, dim=4).permute(0,1,2,4,3)
        x = x.chunk(img_size[0]//self.patch_size[0], dim=3)
        x = torch.cat(x, dim=4).permute(0,1,2,4,3).squeeze(1)
            
        return x     

        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self,):
        return {'pos_embed'}

    def get_head(self,):
        return self.head

    def reset_head(self,):
        self.head = nn.Linear(self.num_features, in_chans*self.patch_size[0]*self.patch_size[1]) 

    def forward_features(self, x, k=None):
        x = self.patch_embed(x)
        _, _, H, W = x.shape
        
        if self.use_pos_embed:
            pos_embed = F.interpolate(self.pos_embed, size=[H, W], mode='bilinear', align_corners = False)
            x = x + pos_embed
            
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for u, blk in enumerate(self.blocks):
            x = blk(x, (H, W))
            if k is not None and u == k:
                self.attention_map = blk.attn.get_attention_map(x, return_map = True)
                
        x = self.norm(x)  

        return x

    # implicit
    # def forward(self, x, k=None):
    #     _, _, H, W = x.shape
    #     x = self.forward_features(x, k)
    #     attention = x
    #     x = self.head(x)
    #     x = self.seq2img(x, (H, W))
        
    #     return attention, x        

    # explicit
    def forward(self, x):
        # input: x = size (1,B*L,K), K = 51+51+1+2 (two obs + action + pos)
        # output: attention = (1,B*L,K), x = (1,B*L,2)

        x = self.forward_features(x)
        attention = x #BxL*Kx9; 9 = num of heads #(B*L)xKx9
        x = self.head(x) #(B*L)xKx1
        x = torch.reshape(x.squeeze(), (int(x.size(0)/self.avrg_img_size), self.avrg_img_size, 105)) #BxLxK
        # x = x.transpose(1,2) # (B*L)x1xK
        # x = torch.reshape(x, (x.size(0), int(x.size(2)/105), 105)) # BxLxK
        x = self.mlp(x) # BxLx2
        
        return attention, x        