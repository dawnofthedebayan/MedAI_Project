import torch
import torch.nn as nn
import torch.nn.functional as F
from .hardnet_68 import hardnet


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResidualBlockSE(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlockSE, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3,stride,padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels,1,padding=0)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        out = self.conv_1x1(out)
        return out

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x




class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel,output_ch=1):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, output_ch, 1)

    def forward(self, x1, x2, x3,x4):
        
        x1_1 = x1

        
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample2(self.upsample(self.upsample(self.upsample(x1)))) * self.conv_upsample3(self.upsample(self.upsample(x2))) * self.conv_upsample3(self.upsample(x3)) * x4
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        
        x4_2 = torch.cat((x4_1, self.conv_upsample6(self.upsample(x3_2))), 1)
        
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x


        #print(x2_rfb.shape,x3_rfb.shape,x4_rfb.shape)
        
class DecoderBlock(nn.Module):
    def __init__(self, in_c, in_c_2, out_c):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_c_2, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlockSE(in_c+out_c, out_c)
        self.r2 = ResidualBlockSE(out_c, out_c)

    def forward(self, x, s):
        
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


class DecoderBlockRA(nn.Module):
    def __init__(self, in_c, in_c_2, out_c):
        super(DecoderBlockRA, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_c_2, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlockSE(in_c+out_c, out_c)
        self.r2 = ResidualBlockSE(out_c, out_c)

    def forward(self, x, s,ra_edge,ra_edge_2):


        x = torch.sigmoid(ra_edge) * x + x
        s = torch.sigmoid(ra_edge_2) * s + s
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


class DecoderBlockMain(nn.Module):

    def __init__(self, in_c, in_c_2, out_c):
        
        super(DecoderBlockMain, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_c_2, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlockSE(in_c+out_c, out_c)
        self.r2 = ResidualBlockSE(out_c, out_c)

        self.r_edge = nn.Sequential(
            ResidualBlock(in_c_2+in_c_2, 1),
            nn.Sigmoid()
        ) 

        self.selayer = SELayer(in_c_2,in_c_2//4)
        


    def forward(self, x, s, edge,ra_seg,ra_edge=None,ra_seg_for_x=None):
        #x4_rfb_seg,x3_rfb_seg,x4_rfb_edge,x_seg_3,x_edge_4,x_seg_4
        #print(x.shape,s.shape,edge.shape)
        #edge = torch.sigmoid(ra_edge) * edge
        
        #edge_x = torch.cat([x, edge], axis=1)
        #edge_x = self.r_edge(edge_x)
        #x = x * edge_x
        if ra_seg_for_x is not None: 
            x = torch.sigmoid(ra_seg_for_x)*x + x
        
        

        edge_x = torch.cat([x, edge], axis=1)
        edge_x = self.r_edge(edge_x)
        ch_attn =  self.selayer(x * edge_x)
        x = x * edge_x + x + ch_attn
        
        if ra_seg is not None: 
            s = torch.sigmoid(ra_seg) * s + s
        
       
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x



class DecoderBlockMainRA(nn.Module):

    def __init__(self, in_c, in_c_2, out_c):
        
        super(DecoderBlockMainRA, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_c_2, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlockSE(in_c+out_c, out_c)
        self.r2 = ResidualBlockSE(out_c, out_c)

        self.r_edge = nn.Sequential(
            ResidualBlock(in_c_2+in_c_2, 1),
            nn.Sigmoid()
        ) 

        self.selayer = SELayer(in_c_2,in_c_2//4)
        


    def forward(self, x, s, edge,ra_seg=None,ra_edge=None,ra_seg_for_x=None):
     
        #print(x.shape,s.shape,edge.shape)
        edge = torch.sigmoid(ra_edge) * edge + edge
        
        #edge_x = torch.cat([x, edge], axis=1)
        #edge_x = self.r_edge(edge_x)
        #x = x * edge_x
        if ra_seg_for_x is not None: 
            x = torch.sigmoid(ra_seg_for_x)*x + x
        
        

        edge_x = torch.cat([x, edge], axis=1)
        edge_x = self.r_edge(edge_x)
        ch_attn =  self.selayer(x * edge_x)
        x = x * edge_x + x + ch_attn
        
        if ra_seg is not None: 
            s = torch.sigmoid(ra_seg) * s + s
        
       
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


class EdgeNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(EdgeNet, self).__init__()
        # ---- ResNet Backbone ----
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.relu = nn.ReLU(True)

        self.rdb_fusion = RDB2(nb_layers = 5,input_dim=channel,growth_rate=channel,sqexcite="True",aspp="True",reduct_ratio=16)
        self.residual_block = ResidualBlock(channel*3, 1, stride=1, downsample=None)

       

        self.residual_block_transform = ResidualBlockSE(channel,channel)

        self.rfb1_1 = RFB(128, channel)
        self.rfb2_1 = RFB(320, channel)
        self.rfb3_1 = RFB(640, channel)
        self.rfb4_1 = RFB(1024, channel)

        self.dense_agg = aggregation(channel)
        self.channel = channel
        self.residual_block_2 = ResidualBlock(channel, 1, stride=1, downsample=None)
        
        self.decoder_block_1 = DecoderBlock(channel,channel,channel//2)
        self.decoder_block_2 = DecoderBlock(channel,channel//2,channel//4)
        self.decoder_block_3 = DecoderBlock(channel,channel//4,channel//8)


        self.decoder_block_main_1 = DecoderBlockMain(channel,channel,channel//2)
        self.decoder_block_main_2 = DecoderBlockMain(channel,channel//2,channel//4)
        self.decoder_block_main_3 = DecoderBlockMain(channel,channel//4,channel//8)


        #Decoder Conv 
        self.dec_conv_1 = BasicConv2d(64, 1, kernel_size=1, padding=0)
        self.dec_conv_2 = BasicConv2d(32, 1, kernel_size=1, padding=0)
        self.dec_conv_3 = BasicConv2d(16, 1, kernel_size=1, padding=0)

   
        self.upsample = nn.ConvTranspose2d(channel//8, 1, kernel_size=4, stride=2, padding=1)

        self.b_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.b_upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hardnet = hardnet(arch=68)
        
        
    def forward(self, x):
        #print("input",x.size())
        
        hardnetout = self.hardnet(x)
        
        x1 = hardnetout[0]
        x2 = hardnetout[1]
        x3 = hardnetout[2]
        x4 = hardnetout[3]

        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        #RDB Fusion for Main Segmentation Task
        x1_rfb_seg = self.residual_block_transform(x1_rfb) + x1_rfb
        x2_rfb_seg = self.residual_block_transform(x2_rfb) + x2_rfb
        x3_rfb_seg = self.residual_block_transform(x3_rfb) + x3_rfb
        x4_rfb_seg = self.residual_block_transform(x4_rfb) + x4_rfb

        #RDB Fusion for Polyp Segmentation Task
        x1_rfb_edge = self.residual_block_transform(x1_rfb) + x1_rfb
        x2_rfb_edge = self.residual_block_transform(x2_rfb) + x2_rfb
        x3_rfb_edge = self.residual_block_transform(x3_rfb) + x3_rfb
        x4_rfb_edge = self.residual_block_transform(x4_rfb) + x4_rfb
        
        #Decoder 1 for Polyp Segmentation
        main_dec_1 = self.decoder_block_main_1(x4_rfb_seg,x3_rfb_seg,x4_rfb_edge,None,None)
        dec_seg_map_1 = self.dec_conv_1(main_dec_1)
        dec_lateral_seg_map_1 = F.interpolate(dec_seg_map_1, scale_factor=16, mode='bilinear') 
        #Decoder 1 for Polyp Edge Segmentation
        edge_dec_1 = self.decoder_block_1(x4_rfb_edge,x3_rfb_edge)
        dec_edge_map_1 = self.dec_conv_1(edge_dec_1)
        dec_lateral_edge_map_1 = F.interpolate(dec_edge_map_1, scale_factor=16, mode='bilinear') 
        
        #Decoder 2 for Polyp Segmentation
        main_dec_2 = self.decoder_block_main_2(main_dec_1,x2_rfb_seg,edge_dec_1,None)
        dec_seg_map_2 = self.dec_conv_2(main_dec_2)
        dec_lateral_seg_map_2 = F.interpolate(dec_seg_map_2, scale_factor=8, mode='bilinear') 
        
        #Decoder 2 for Polyp Edge Segmentation
        edge_dec_2 = self.decoder_block_2(edge_dec_1,x2_rfb_edge)
        dec_edge_map_2 = self.dec_conv_2(edge_dec_2)
        dec_lateral_edge_map_2 = F.interpolate(dec_edge_map_2, scale_factor=8, mode='bilinear') 

        #Decoder 3 for Polyp Segmentation
        main_dec_3 = self.decoder_block_main_3(main_dec_2,x1_rfb_seg,edge_dec_2,None)
        dec_seg_map_3 = self.dec_conv_3(main_dec_3)
        dec_lateral_seg_map_3 = F.interpolate(dec_seg_map_3, scale_factor=4, mode='bilinear') 

        #Decoder 3 for Polyp Edge Segmentation
        edge_dec_3 = self.decoder_block_3(edge_dec_2,x1_rfb_edge)
        dec_edge_map_3 = self.dec_conv_3(edge_dec_3)
        dec_lateral_edge_map_3 = F.interpolate(dec_edge_map_3, scale_factor=4, mode='bilinear') 

        seg_map = self.b_upsample(self.upsample(main_dec_3))
        edge_map = self.b_upsample(self.upsample(edge_dec_3))

        
        return [seg_map,dec_lateral_seg_map_1,dec_lateral_seg_map_2,dec_lateral_seg_map_3],[edge_map,dec_lateral_edge_map_1,dec_lateral_edge_map_2,dec_lateral_edge_map_3]



class DRPAEdgeNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(DRPAEdgeNet, self).__init__()
        # ---- ResNet Backbone ----
        #self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.relu = nn.ReLU(True)

        self.residual_block = ResidualBlock(channel*3, 1, stride=1, downsample=None)

       

        self.residual_block_transform = ResidualBlockSE(channel,channel)

        self.rfb1_1 = RFB(128, channel)
        self.rfb2_1 = RFB(320, channel)
        self.rfb3_1 = RFB(640, channel)
        self.rfb4_1 = RFB(1024, channel)

        self.dense_agg = aggregation(channel)
        self.channel = channel
        self.residual_block_2 = ResidualBlock(channel, 1, stride=1, downsample=None)
        
        self.decoder_block_1 = DecoderBlockRA(channel,channel,channel//2)
        self.decoder_block_2 = DecoderBlockRA(channel,channel//2,channel//4)
        self.decoder_block_3 = DecoderBlockRA(channel,channel//4,channel//8)


        self.decoder_block_main_1 = DecoderBlockMainRA(channel,channel,channel//2)
        self.decoder_block_main_2 = DecoderBlockMainRA(channel,channel//2,channel//4)
        self.decoder_block_main_3 = DecoderBlockMainRA(channel,channel//4,channel//8)

       
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(320, 32, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(32, 1, kernel_size=3, padding=1)

        # ---- reverse attention branch 1 ----
        self.ra1_conv1 = BasicConv2d(128, 32, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(32, 1, kernel_size=3, padding=1)
        

        #Decoder Conv 
        self.dec_conv_1 = BasicConv2d(64, 1, kernel_size=1, padding=0)
        self.dec_conv_2 = BasicConv2d(32, 1, kernel_size=1, padding=0)
        self.dec_conv_3 = BasicConv2d(16, 1, kernel_size=1, padding=0)

   
        self.upsample = nn.ConvTranspose2d(channel//8, 1, kernel_size=4, stride=2, padding=1)

        self.b_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.b_upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hardnet = hardnet(arch=68)
        
        
    def forward(self, x):
        #print("input",x.size())
        
        hardnetout = self.hardnet(x)
        
        x1 = hardnetout[0]
        x2 = hardnetout[1]
        x3 = hardnetout[2]
        x4 = hardnetout[3]

        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        #RDB Fusion for Main Segmentation Task
        #x1_rfb_seg,x2_rfb_seg,x3_rfb_seg,x4_rfb_seg = self.rdb_fusion(x1_rfb,x2_rfb,x3_rfb,x4_rfb)
        x1_rfb_seg = self.residual_block_transform(x1_rfb) + x1_rfb
        x2_rfb_seg = self.residual_block_transform(x2_rfb) + x2_rfb
        x3_rfb_seg = self.residual_block_transform(x3_rfb) + x3_rfb
        x4_rfb_seg = self.residual_block_transform(x4_rfb) + x4_rfb

        #RDB Fusion for Polyp Segmentation Task
        #x1_rfb_edge,x2_rfb_edge,x3_rfb_edge,x4_rfb_edge = self.rdb_fusion(x1_rfb,x2_rfb,x3_rfb,x4_rfb)
        x1_rfb_edge = self.residual_block_transform(x1_rfb) + x1_rfb
        x2_rfb_edge = self.residual_block_transform(x2_rfb) + x2_rfb
        x3_rfb_edge = self.residual_block_transform(x3_rfb) + x3_rfb
        x4_rfb_edge = self.residual_block_transform(x4_rfb) + x4_rfb
        
        #Crude Segmentation for Polyp Segmentation Task
        crude_seg_feat = self.dense_agg(x4_rfb_seg,x3_rfb_seg,x2_rfb_seg,x1_rfb_seg)
        crude_seg_map  = self.b_upsample_4x(crude_seg_feat)

        #Crude Segmentation for Edge Segmentation Task
        crude_edge_feat = self.dense_agg(x4_rfb_edge,x3_rfb_edge,x2_rfb_edge,x1_rfb_edge)
        crude_edge_map  = self.b_upsample_4x(crude_edge_feat)

        #Reverse Attention Polyp Segmentation 
        #branch 4 
        crop_seg_4 = F.interpolate(crude_seg_feat, scale_factor=0.125, mode='bilinear')
        x_seg = -1*(torch.sigmoid(crop_seg_4)) + 1
        x_seg = x_seg.expand(-1, 1024, -1, -1).mul(x4)
        x_seg = self.ra4_conv1(x_seg)
        x_seg = F.relu(self.ra4_conv2(x_seg))
        x_seg = F.relu(self.ra4_conv3(x_seg))
        x_seg = F.relu(self.ra4_conv4(x_seg))
        x_seg = F.relu(self.ra4_conv5(x_seg))
        x_seg_4 = x_seg + crop_seg_4

        lateral_seg_map_4 = F.interpolate(x_seg_4, scale_factor=32, mode='bilinear') # (bs, 1, 8, 8) -> (bs, 1, 256, 256)

        #branch 3 
        crop_seg_3 = F.interpolate(x_seg_4, scale_factor=2, mode='bilinear')
        x_seg = -1*(torch.sigmoid(crop_seg_3)) + 1
        x_seg = x_seg.expand(-1, 640, -1, -1).mul(x3)
        x_seg = self.ra3_conv1(x_seg)
        x_seg = F.relu(self.ra3_conv2(x_seg))
        x_seg = F.relu(self.ra3_conv3(x_seg))
        x_seg = F.relu(self.ra3_conv4(x_seg))
        x_seg_3 = x_seg + crop_seg_3

        lateral_seg_map_3 = F.interpolate(x_seg_3, scale_factor=16, mode='bilinear') # (bs, 1, 16, 16) -> (bs, 1, 256, 256)


        #branch 2 
        crop_seg_2 = F.interpolate(x_seg_3, scale_factor=2, mode='bilinear')
        x_seg = -1*(torch.sigmoid(crop_seg_2)) + 1
        x_seg = x_seg.expand(-1, 320, -1, -1).mul(x2)
        x_seg = self.ra2_conv1(x_seg)
        x_seg = F.relu(self.ra2_conv2(x_seg))
        x_seg = F.relu(self.ra2_conv3(x_seg))
        x_seg = F.relu(self.ra2_conv4(x_seg))
        x_seg_2 = x_seg + crop_seg_2

        lateral_seg_map_2 = F.interpolate(x_seg_2, scale_factor=8, mode='bilinear') # (bs, 1, 32, 32) -> (bs, 1, 256, 256)


        #branch 1 
        crop_seg_1 = F.interpolate(x_seg_2, scale_factor=2, mode='bilinear')
        x_seg = -1*(torch.sigmoid(crop_seg_1)) + 1
        x_seg = x_seg.expand(-1, 128, -1, -1).mul(x1)
        x_seg = self.ra1_conv1(x_seg)
        x_seg = F.relu(self.ra1_conv2(x_seg))
        x_seg = F.relu(self.ra1_conv3(x_seg))
        x_seg = F.relu(self.ra1_conv4(x_seg))
        x_seg_1 = x_seg + crop_seg_1

        lateral_seg_map_1 = F.interpolate(x_seg_1, scale_factor=4, mode='bilinear') # (bs, 1, 32, 32) -> (bs, 1, 256, 256)
        
        #Reverse Attention Edge Segmentation 
        #branch 4
        crop_edge_4 = F.interpolate(crude_edge_feat, scale_factor=0.125, mode='bilinear')
        x_edge = -1*(torch.sigmoid(crop_edge_4)) + 1
        x_edge = x_edge.expand(-1, 1024, -1, -1).mul(x4)
        x_edge = self.ra4_conv1(x_edge)
        x_edge = F.relu(self.ra4_conv2(x_edge))
        x_edge = F.relu(self.ra4_conv3(x_edge))
        x_edge = F.relu(self.ra4_conv4(x_edge))
        x_edge = F.relu(self.ra4_conv5(x_edge))
        x_edge_4 = x_edge + crop_edge_4

        lateral_edge_map_4 = F.interpolate(x_edge_4, scale_factor=32, mode='bilinear') # (bs, 1, 8, 8) -> (bs, 1, 256, 256)
       
        #branch 3
        crop_edge_3 = F.interpolate(x_edge_4, scale_factor=2, mode='bilinear')
        x_edge = -1*(torch.sigmoid(crop_edge_3)) + 1
        x_edge = x_edge.expand(-1, 640, -1, -1).mul(x3)
        x_edge = self.ra3_conv1(x_edge)
        x_edge = F.relu(self.ra3_conv2(x_edge))
        x_edge = F.relu(self.ra3_conv3(x_edge))
        x_edge = F.relu(self.ra3_conv4(x_edge))
        x_edge_3 = x_edge + crop_edge_3

        lateral_edge_map_3 = F.interpolate(x_edge_3, scale_factor=16, mode='bilinear') # (bs, 1, 16, 16) -> (bs, 1, 256, 256)


        #branch 2
        crop_edge_2 = F.interpolate(x_edge_3, scale_factor=2, mode='bilinear')
        x_edge = -1*(torch.sigmoid(crop_edge_2)) + 1
        x_edge = x_edge.expand(-1, 320, -1, -1).mul(x2)
        x_edge = self.ra2_conv1(x_edge)
        x_edge = F.relu(self.ra2_conv2(x_edge))
        x_edge = F.relu(self.ra2_conv3(x_edge))
        x_edge = F.relu(self.ra2_conv4(x_edge))
        x_edge_2 = x_edge + crop_edge_2

        lateral_edge_map_2 = F.interpolate(x_edge_2, scale_factor=8, mode='bilinear') # (bs, 1, 32, 32) -> (bs, 1, 256, 256)


        #branch 1
        crop_edge_1 = F.interpolate(x_edge_2, scale_factor=2, mode='bilinear')
        x_edge = -1*(torch.sigmoid(crop_edge_1)) + 1
        x_edge = x_edge.expand(-1, 128, -1, -1).mul(x1)
        x_edge = self.ra1_conv1(x_edge)
        x_edge = F.relu(self.ra1_conv2(x_edge))
        x_edge = F.relu(self.ra1_conv3(x_edge))
        x_edge = F.relu(self.ra1_conv4(x_edge))
        x_edge_1 = x_edge + crop_edge_1

        lateral_edge_map_1 = F.interpolate(x_edge_1, scale_factor=4, mode='bilinear') # (bs, 1, 64, 64) -> (bs, 1, 256, 256)
       
        
        #Decoder 1 for Polyp Segmentation
        main_dec_1 = self.decoder_block_main_1(x4_rfb_seg,x3_rfb_seg,x4_rfb_edge,x_seg_3,x_edge_4,x_seg_4)
        dec_seg_map_1 = self.dec_conv_1(main_dec_1)
        dec_lateral_seg_map_1 = F.interpolate(dec_seg_map_1, scale_factor=16, mode='bilinear') 

        #Decoder 1 for Polyp Edge Segmentation
        edge_dec_1 = self.decoder_block_1(x4_rfb_edge,x3_rfb_edge,x_edge_4,x_edge_3)
        dec_edge_map_1 = self.dec_conv_1(edge_dec_1)
        dec_lateral_edge_map_1 = F.interpolate(dec_edge_map_1, scale_factor=16, mode='bilinear') 
        
        #Decoder 2 for Polyp Segmentation
        main_dec_2 = self.decoder_block_main_2(main_dec_1,x2_rfb_seg,edge_dec_1,x_seg_2,x_edge_3,x_seg_3)
        dec_seg_map_2 = self.dec_conv_2(main_dec_2)
        dec_lateral_seg_map_2 = F.interpolate(dec_seg_map_2, scale_factor=8, mode='bilinear') 
        
        #Decoder 2 for Polyp Edge Segmentation
        edge_dec_2 = self.decoder_block_2(edge_dec_1,x2_rfb_edge,x_edge_3,x_edge_2)
        dec_edge_map_2 = self.dec_conv_2(edge_dec_2)
        dec_lateral_edge_map_2 = F.interpolate(dec_edge_map_2, scale_factor=8, mode='bilinear') 

        #Decoder 3 for Polyp Segmentation
        main_dec_3 = self.decoder_block_main_3(main_dec_2,x1_rfb_seg,edge_dec_2,x_seg_1,x_edge_2,x_seg_2)
        dec_seg_map_3 = self.dec_conv_3(main_dec_3)
        dec_lateral_seg_map_3 = F.interpolate(dec_seg_map_3, scale_factor=4, mode='bilinear') 

        #Decoder 3 for Polyp Edge Segmentation
        edge_dec_3 = self.decoder_block_3(edge_dec_2,x1_rfb_edge,x_edge_2,x_edge_1)
        dec_edge_map_3 = self.dec_conv_3(edge_dec_3)
        dec_lateral_edge_map_3 = F.interpolate(dec_edge_map_3, scale_factor=4, mode='bilinear') 

        seg_map = self.b_upsample(self.upsample(main_dec_3))
        edge_map = self.b_upsample(self.upsample(edge_dec_3))

        
        return [seg_map,crude_seg_map,dec_lateral_seg_map_1,dec_lateral_seg_map_2,dec_lateral_seg_map_3,lateral_seg_map_1,lateral_seg_map_2,lateral_seg_map_3,lateral_seg_map_4],\
        [edge_map,crude_edge_map,dec_lateral_edge_map_1,dec_lateral_edge_map_2,dec_lateral_edge_map_3,lateral_edge_map_1,lateral_edge_map_2,lateral_edge_map_3,lateral_edge_map_4]



