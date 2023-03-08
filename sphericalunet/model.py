#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:30 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import torch
import torch.nn as nn
from .utils.utils import Get_neighs_order, Get_upconv_index, Get_swin_matrices_2order
from .layers import onering_conv_layer, pool_layer, upconv_layer, pool_layer_batch, upconv_layer_batch, self_attention_layer_swin



class down_block(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first = False):
        super(down_block, self).__init__()

        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
        )
            
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders, 'mean'),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        
        return x


class up_block(nn.Module):
    """Define the upsamping block in spherica unet
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block, self).__init__()
        
        self.up = upconv_layer(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1)
        x = self.double_conv(x)

        return x
    

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class patch_merging_layer(nn.Module):
    def __init__(self, neigh_orders, in_ch, out_ch):
        super(patch_merging_layer, self).__init__()

        self.neigh_orders = neigh_orders
        self.proj = nn.Conv1d(int(in_ch*7), out_ch, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False)
        # self.norm = nn.BatchNorm1d(out_ch, momentum=0.15, affine=False, track_running_stats=True)
        # self.norm = nn.InstanceNorm1d(out_ch, momentum=0.15)
    
    def forward(self, x):
        batch_num, feat_num, num_nodes = x.shape
        num_nodes = int((x.size()[2]+6)/4)
        feat_num = x.size()[1]
        x = x[:, :, self.neigh_orders[0:num_nodes*7]].view(batch_num, feat_num, num_nodes, 7).permute((0, 3, 1, 2)).reshape((batch_num, -1, num_nodes))
        x = self.norm(self.proj(x))
        return x


class down_block_batch(nn.Module):
    """
    downsampling block in spherical unet
    mean pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False, drop_rate=None, num_heads=1):
        super(down_block_batch, self).__init__()
        self.first = first
        # Batch norm version
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                # nn.BatchNorm1d(out_ch, momentum=0.15, affine=False, track_running_stats=True),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.InstanceNorm1d(out_ch, momentum=0.15),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                # nn.BatchNorm1d(out_ch, momentum=0.15, affine=False, track_running_stats=True),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.InstanceNorm1d(out_ch, momentum=0.15),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False)
        )

            
        else:
            self.block = nn.Sequential(
                pool_layer_batch(pool_neigh_orders),
                conv_layer(in_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
                conv_layer(out_ch, out_ch, neigh_orders, drop_rate=drop_rate),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                # nn.GroupNorm(group_num, out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=False),
        )


    def forward(self, x):
        # batch norm version
        x = self.block(x)
        return x


class up_block_batch(nn.Module):
    """Define the upsamping block in spherica uent
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            neigh_orders (tensor, int)  - - conv layer's filters' neighborhood orders
            
    """    
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, upconv_top_index, upconv_down_index):
        super(up_block_batch, self).__init__()
        
        self.up = upconv_layer_batch(in_ch, out_ch, upconv_top_index, upconv_down_index)
        
        # batch norm version
        self.double_conv = nn.Sequential(
             conv_layer(in_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True),
             conv_layer(out_ch, out_ch, neigh_orders),
             nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
             nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        # x = torch.cat((x1, x2), 2) 
        # x = self.double_conv(x.permute(0,2,1)).permute(0,2,1)
        x = torch.cat((x1, x2), 1) 
        x = self.double_conv(x)

        return x


class SUnet(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch, level=7, n_res=5, rotated=0, complex_chs=16):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
            level (int) - - input surface's icosahedron level. default: 7, for 40962 vertices
                            2:42, 3:162, 4:642, 5:2562, 6:10242
            n_res (int) - - the total resolution levels of u-net, default: 5
            rotated (int) - -  the sphere is original (0), rotated 90 degrees along y axis (0), or 
                               90 degrees along x axis (1)
        """
        super(SUnet, self).__init__()
        
        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than input level"
        assert n_res >=2, "number of resolution levels should be larger than 2"     
        assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
        
        neigh_orders = Get_neighs_order(rotated)
        # import pdb
        # pdb.set_trace()
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index(rotated)
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*complex_chs)
        conv_layer = onering_conv_layer
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
      
        self.up = nn.ModuleList([])
        for i in range(n_res-1):
            self.up.append(up_block(conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
            
        self.outc = nn.Linear(chs[1], out_ch)
                
        self.n_res = n_res
        
    def forward(self, x):
        # x's size should be [N (number of vertices) x C (channel)]
        xs = [x]
        for i in range(self.n_res):
            xs.append(self.down[i](xs[i]))

        x = xs[-1]
        for i in range(self.n_res-1):
            x = self.up[i](x, xs[self.n_res-1-i])

        x = self.outc(x)
        return x
        
    

class GenAgeNet(nn.Module):
    """Generation model for atlas construction

    """    
    def __init__(self, level=6, gender=False, out_ch=2):
        """ Initialize the model.

        Parameters:
            n_sub (int) - -  number of the subjects in the group
            level (int) - -  The generated atlas level, default 6 with 10242 vertices
            age (bool) - -   add variable age?
            gender (bool) - -add variable gender? 
        """
        super(GenAgeNet, self).__init__()
        
        self.gender = gender
        self.level = level
        # self.n_sub = n_sub
        
        neigh_orders = Get_neighs_order(rotated=0)
        neigh_orders = neigh_orders[8-level:]
        upconv_index = Get_upconv_index(rotated=0)[(8-level)*2:4]
        
        n_vertex = int(len(neigh_orders[0])/7)
        assert n_vertex in [42,642,2562,10242,40962,163842]
        self.n_vertex = n_vertex

        self.fc_age = nn.Linear(1, 256)
        
        if gender is False:
            chs_0 = 256
        elif gender is True:
            chs_0 = 258  # add variable gender here
        else:
            raise NotImplementedError('Not implemented.')
        
        chs = [3, 8, 8, out_ch]
        if level <= 6:
            self.fc = nn.Linear(chs_0, chs[0]*n_vertex)
        else:
            self.fc = nn.Linear(chs_0, chs[0]*10242)
        
        if level > 6 :
            upblock_list = []
            for i in range(level-6):
                upblock_list.append(nn.BatchNorm1d(chs[0], momentum=0.15, affine=True, track_running_stats=False))
                upblock_list.append(nn.LeakyReLU(0.2))
                upblock_list.append(upconv_layer(chs[0], chs[0], upconv_index[-i*2-2], upconv_index[-i*2-1]))
            self.upconv = torch.nn.Sequential(*upblock_list)
    
        conv_list = []
        for i in range(len(chs)-1):
            conv_list.append(nn.BatchNorm1d(chs[i], momentum=0.15, affine=True, track_running_stats=False))
            conv_list.append(nn.LeakyReLU(0.2))
            conv_list.append(onering_conv_layer(chs[i], chs[i+1], neigh_orders[0]))
        self.conv_block = torch.nn.Sequential(*conv_list)
        
    def forward(self, age=0, gender=0):
        # assert sub_id.shape == torch.Size([1, self.n_sub])
        # x_sub = self.fc_sub(sub_id)      # 1*1024
        assert age.shape == torch.Size([1, 1])
        x_age = self.fc_age(age)     # 1*256
        if self.gender:
            assert gender.shape == torch.Size([1, 2])
            x = torch.cat((x_age, gender),1)   # 1*2050
        else:
            x = x_age
            
        x = self.fc(x) # 1* (10242*3)
        if self.n_vertex <= 10242:
            x = torch.reshape(x, (self.n_vertex,-1)) # 10242 * 3
        else:
            x = torch.reshape(x, (10242,-1))  # 10242 * 3
            x = self.upconv(x)
            
        x = self.conv_block(x)
        
        return x
    

class GenPhiUsingSubId(nn.Module):
    """Generating deformation field from atlas to within-subject-mean

    """    
    def __init__(self, level, n_sub):
        """ Initialize the model.

        Parameters:
            n_sub (int) - -  number of the subjects in the group
            level (int) - -  The generated atlas level, default 6 with 10242 vertices
            age (bool) - -   add variable age?
            gender (bool) - -add variable gender? 
        """
        super(GenPhiUsingSubId, self).__init__()
        
        self.level = level
        self.n_sub = n_sub
        
        neigh_orders = Get_neighs_order(rotated=0)
        neigh_orders = neigh_orders[8-level:]
        upconv_index = Get_upconv_index(rotated=0)[(8-level)*2:4]
        
        n_vertex = int(len(neigh_orders[0])/7)
        assert n_vertex in [42,642,2562,10242,40962,163842]
        self.n_vertex = n_vertex

        self.fc_sub = nn.Linear(n_sub, 256)
        
        chs_0 = 256
        
        chs = [3, 8, 8, 2]
        if level <= 6:
            self.fc = nn.Linear(chs_0, chs[0]*n_vertex)
        else:
            self.fc = nn.Linear(chs_0, chs[0]*10242)
        
        if level > 6 :
            upblock_list = []
            for i in range(level-6):
                upblock_list.append(nn.BatchNorm1d(chs[0], momentum=0.15, affine=True, track_running_stats=False))
                upblock_list.append(nn.LeakyReLU(0.2))
                upblock_list.append(upconv_layer(chs[0], chs[0], upconv_index[-i*2-2], upconv_index[-i*2-1]))
            self.upconv = torch.nn.Sequential(*upblock_list)
    
        conv_list = []
        for i in range(len(chs)-1):
            conv_list.append(nn.BatchNorm1d(chs[i], momentum=0.15, affine=True, track_running_stats=False))
            conv_list.append(nn.LeakyReLU(0.2))
            conv_list.append(onering_conv_layer(chs[i], chs[i+1], neigh_orders[0]))
        self.conv_block = torch.nn.Sequential(*conv_list)
        
    def forward(self, sub_id):
        assert sub_id.shape == torch.Size([1, self.n_sub])
        x = self.fc_sub(sub_id)      # 1*1024
        x = self.fc(x) # 1* (10242*3)
        if self.n_vertex <= 10242:
            x = torch.reshape(x, (self.n_vertex,-1)) # 10242 * 3
        else:
            x = torch.reshape(x, (10242,-1))  # 10242 * 3
            x = self.upconv(x)
            
        x = self.conv_block(x)
        
        return x
        

class UNet18_10k_SWIN_pred(nn.Module):
    """Define the Spherical UNet structure

    """    
    def __init__(self, in_ch, out_ch, drop_rate):
        """ Initialize the Spherical UNet.

        Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels
        """
        super(UNet18_10k_SWIN_pred, self).__init__()
        
        neigh_orders = Get_neighs_order()[1:]
        matrices = Get_swin_matrices_2order()[1:] # For 2-ring neighborhood
        # matrices = Get_swin_matrices_4order()[1:] # For 4-ring neighborhood
        # matrices = Get_neighs_order()[1:] # For 1-ring neighborhood

        upconv_index = Get_upconv_index()

        # chs = [64, 128, 256, 512]
        # chs = [32, 32, 64, 64, 128]
        chs = [16, 32, 64, 128]

        conv_layer = self_attention_layer_swin # For 2-ring neighborhood
        # conv_layer = self_attention_layer_swin_4order # For 4-ring neighborhood
        # conv_layer = self_attention_layer_batch # For 1-ring neighborhood

        self.init = nn.Conv1d(in_ch, chs[0], kernel_size=1)
        self.down1 = down_block_batch(conv_layer, chs[0], chs[0], matrices[0], None, True, drop_rate=drop_rate)
        self.patch_merging_2 = patch_merging_layer(neigh_orders[1], chs[0], chs[1])

        self.down2 = down_block_batch(conv_layer, chs[1], chs[1], matrices[1], None, True, drop_rate=drop_rate)
        self.patch_merging_3 = patch_merging_layer(neigh_orders[2], chs[1], chs[2])

        self.down3 = down_block_batch(conv_layer, chs[2], chs[2], matrices[2], None, True, drop_rate=drop_rate)
        self.patch_merging_4 = patch_merging_layer(neigh_orders[3], chs[2], chs[3])

        self.down4 = down_block_batch(conv_layer, chs[3], chs[3], matrices[3], None, True, drop_rate=drop_rate)

        self.upsample_1 = upconv_layer_batch(chs[3], chs[2], upconv_index[-4], upconv_index[-3])
        self.up1 = down_block_batch(conv_layer, int(chs[2]*2), chs[2], matrices[2], None, True, drop_rate=drop_rate)

        self.upsample_2 = upconv_layer_batch(chs[2], chs[1], upconv_index[-6], upconv_index[-5])
        self.up2 = down_block_batch(conv_layer, int(chs[1]*2), chs[1], matrices[1], None, True, drop_rate=drop_rate)

        self.upsample_3 = upconv_layer_batch(chs[1], chs[0], upconv_index[-8], upconv_index[-7])
        self.up3 = down_block_batch(conv_layer, int(chs[0]*2), chs[0], matrices[0], None, True, drop_rate=drop_rate)

        # self.pool2 = nn.AdaptiveAvgPool1d((1))

        # self.rnn = nn.LSTM(in_ch, chs[0]//2, num_layers=1, bidirectional=True)
        # self.rnn = nn.GRU(in_ch, chs[0], num_layers=1)
        # self.rnn = nn.RNN(in_ch, chs[0], num_layers=1)

        # self.rnn.flatten_parameters()
        self.outc = nn.Sequential(
                nn.Conv1d(chs[0], out_ch, kernel_size=1)
                )
        self.apply(weight_init)
        self.grads = {}
    
    def my_hook(self, module, grad_input, grad_output):
        # print('original grad:', grad_input)
        # print('original outgrad:', grad_output)     

        # import pdb
        # pdb.set_trace()
        self.grads['x9'] = grad_output[0]
        return grad_input
                
        
    def forward(self, x):
        B, N, C = x.shape
        x1 = torch.Tensor.permute(x, (0, 2, 1))

        x1 = self.init(x1)
        x1_1 = self.down1(x1)
        x2 = self.patch_merging_2(x1+x1_1)
        x2_1 = self.down2(x2)
        x3 = self.patch_merging_3(x2+x2_1)
        x3_1 = self.down3(x3)
        x4 = self.patch_merging_4(x3+x3_1)
        x4_1 = self.down4(x4)

        x = self.up1(torch.cat((self.upsample_1(x4_1), x3_1), dim=1))
        x = self.up2(torch.cat((self.upsample_2(x), x2_1), dim=1))
        x = self.up3(torch.cat((self.upsample_3(x), x1_1), dim=1))

        x = self.outc(x)

        return x



if __name__ == "__main__":
    test_model = Unet(in_ch=3, out_ch=64)
    print("True")
