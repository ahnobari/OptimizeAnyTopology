from .layers import *
from .layers import BC_Encoder, C_Encoder

class Encoder(nn.Module):
    def __init__(self, hidden_channels=128, ch_mult=(1,2,4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=256, z_channels=3, **ignore_kwargs):
        super().__init__()
        self.ch = hidden_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch*in_ch_mult[i_level]
            block_out = self.ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        self.final_conv = torch.nn.Conv2d(z_channels,
                                          z_channels,
                                          1)
    def forward(self, x):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        h = self.final_conv(h)
        return h
    
class Decoder(nn.Module):
    def __init__(self, *, hidden_channels=128, ch_mult=(1,2,4), num_res_blocks=2,
                 dropout=0.0, resamp_with_conv=True,
                 resolution=256, z_channels=3, **ignorekwargs):
        super().__init__()
        self.ch = hidden_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        self.init_conv = torch.nn.Conv2d(z_channels,z_channels,1)
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

    def forward(self, z):
        self.last_z_shape = z.shape

        h = self.init_conv(z)
        
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)


        return h
    
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super().__init__()

        self.model = NLayerDiscriminator(input_nc=input_nc)
        self.model.apply(self.weights_init)
    
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.model(x)

class SimpleRendererNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=3, kernel_size=3, normalize=True, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2),
                                    ResnetBlock(in_channels=hidden_channels,
                                                out_channels=hidden_channels,
                                                kernel_size=kernel_size,
                                                temb_channels=0, dropout=0.0, normalize=normalize),
                                    ResnetBlock(in_channels=hidden_channels,
                                                out_channels=hidden_channels,
                                                kernel_size=kernel_size,
                                                temb_channels=0, dropout=0.0, normalize=normalize)])
        self.norm_out = Normalize(hidden_channels) if normalize else torch.nn.Identity()
        self.conv_out = torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def get_last_layer_weight(self):
        return self.conv_out.weight

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1, 2]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x
    
class ConcatWrapper(nn.Module):
    def __init__(self, z_dec_channels, co_pe_dim=None, co_pe_w_max=None, ce_pe_dim=None, ce_pe_w_max=None, x_channels=None, hidden_channels=256, out_channels=3, kernel_size=3, normalize=True, out_act='tanh', *args, **kwargs):
        super().__init__(),
        self.x_channels = x_channels
        coord_dim = 0

        self.co_pe_dim = co_pe_dim
        self.co_pe_w_max = co_pe_w_max
        coord_dim += 2 if co_pe_dim is None else 2 * co_pe_dim

        self.ce_pe_dim = ce_pe_dim
        self.ce_pe_w_max = ce_pe_w_max
        coord_dim += 2 if ce_pe_dim is None else 2 * ce_pe_dim

        in_ch = (x_channels if x_channels is not None else 0) + z_dec_channels + coord_dim
        self.net = SimpleRendererNet(in_channels=in_ch, hidden_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, normalize=normalize)

        self.out_act = out_act

    def get_last_layer_weight(self):
        return self.net.get_last_layer_weight()

    def forward(self, z_dec, coord, cell):
        if isinstance(coord, list):
            out = []
            for i in range(len(coord)):
                q_feat, rel_coord, rel_cell = convert_liif_feat_coord_cell(z_dec[i:i+1], coord[i], cell[i])
                if self.co_pe_dim is not None:
                    rel_coord = convert_posenc(rel_coord, self.co_pe_dim, self.co_pe_w_max)
                if self.ce_pe_dim is not None:
                    rel_cell = convert_posenc(rel_cell, self.ce_pe_dim, self.ce_pe_w_max)

                # mask out the padding
                layout = torch.cat([q_feat, rel_coord, rel_cell], dim=-1).permute(0, 3, 1, 2)
                o = self.net(layout)
                if self.out_act == 'tanh':
                    o = torch.tanh(o)
                elif self.out_act == 'sigmoid':
                    o = torch.sigmoid(o)
                out.append(o)
                
        else:
            q_feat, rel_coord, rel_cell = convert_liif_feat_coord_cell(z_dec, coord, cell)
            if self.co_pe_dim is not None:
                rel_coord = convert_posenc(rel_coord, self.co_pe_dim, self.co_pe_w_max)
            if self.ce_pe_dim is not None:
                rel_cell = convert_posenc(rel_cell, self.ce_pe_dim, self.ce_pe_w_max)
            
            # mask out the padding
            layout = torch.cat([q_feat, rel_coord, rel_cell], dim=-1).permute(0, 3, 1, 2)
            
            out = self.net(layout)
            
            if self.out_act == 'tanh':
                out = torch.tanh(out)
            elif self.out_act == 'sigmoid':
                out = torch.sigmoid(out)
        
        return out


class ProblemEncoder(nn.Module):
    def __init__(self,
                 BCs = [4,4],
                 BC_n_layers = [4,4],
                 BC_hidden_size = [256,256], 
                 BC_emb_size = [64,64], 
                 Cs = [1,2],
                 C_n_layers = [4,4],
                 C_hidden_size = [256,256],
                 C_mapping_size = [128,128],
                 latent_size = 256,
                 **ignore_kwargs):
        
        super(ProblemEncoder, self).__init__()
        
        # check if BC_n_layers and BC_hidden_size are lists of same length
        assert len(BC_n_layers) == len(BC_hidden_size)
        assert len(BC_n_layers) == len(BC_emb_size)
        assert len(BC_n_layers) == len(BCs)
        
        # check if C_mapping_size is a list of same length as Cs
        assert len(C_n_layers) == len(Cs)
        assert len(C_n_layers) == len(C_mapping_size)
        assert len(C_n_layers) == len(C_hidden_size)
        
        latent_conditional_dim = 0
        
        if len(BCs) > 0:
            self.hasBC = True
            self.BC_Networks = nn.ModuleList()
            for i in range(len(BCs)):
                self.BC_Networks.append(BC_Encoder([BCs[i]] + [BC_hidden_size[i]]* BC_n_layers[i] + [BC_emb_size[i]]))
                latent_conditional_dim += BC_emb_size[i]*3
        else:
            self.hasBC = False
        
        if len(Cs) > 0:
            self.hasC = True
            self.C_Networks = nn.ModuleList()
            for i in range(len(Cs)):
                self.C_Networks.append(C_Encoder([Cs[i]] + [C_hidden_size[i]]* C_n_layers[i] + [C_mapping_size[i]]))
                latent_conditional_dim += C_mapping_size[i]
        else:
            self.hasC = False
            
        self.latent_conditional_dim = latent_conditional_dim
        
        self.linear1 = nn.Linear(latent_conditional_dim, latent_size)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(latent_size, latent_size)
        
        if not self.hasC and not self.hasBC:
            raise ValueError("Error: ProblemEncoder must have at least one BC or C")
        
    @torch.compiler.disable
    def forward(self, BCs=None, BC_Batch=None, Cs=None, **kwargs):
        
        if self.hasC:
            bs = Cs[0].shape[0]
            device = Cs[0].device
        else:
            bs = int(BC_Batch[0].max().item() + 1)
            device = BC_Batch[0].device
        
        current_idx = 0
        full_cond = torch.zeros(bs, self.latent_conditional_dim, device=device)
        
        if self.hasBC:
            BC_emb = []
            for i in range(len(BCs)):
                BC_emb.append(self.BC_Networks[i](BCs[i],BC_Batch[i]))
            
            BC_emb = torch.cat(BC_emb,-1)
            full_cond[:,current_idx:current_idx+BC_emb.shape[1]] = BC_emb
            current_idx += BC_emb.shape[1]
        
        if self.hasC:
            C_emb = []
            for i in range(len(Cs)):
                C_emb.append(self.C_Networks[i](Cs[i]))
            
            C_emb = torch.cat(C_emb,-1)
            full_cond[:,current_idx:current_idx+C_emb.shape[1]] = C_emb
            current_idx += C_emb.shape[1]
            
        out = self.linear1(full_cond)
        out = self.act(out)
        out = self.linear2(out)
        
        return out