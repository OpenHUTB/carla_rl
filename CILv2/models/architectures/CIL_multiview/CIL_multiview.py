import torch
import torch.nn as nn
import importlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from configs import g_conf
from CILv2.models.building_blocks import FC
from CILv2.models.building_blocks.PositionalEncoding import PositionalEncoding
from CILv2.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from CILv2.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer


class CIL_multiview(nn.Module):
    def __init__(self, params):
        super(CIL_multiview, self).__init__()
        self.params = params
        self.device = torch.device("cpu")
        resnet_module = importlib.import_module('CILv2.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf.IMAGENET_PRE_TRAINED,
                                                          layer_id = params['encoder_embedding']['perception']['res'][ 'layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf.BATCH_SIZE] + g_conf.IMAGE_SHAPE)[params['encoder_embedding']['perception']['res'][ 'layer_id']]

        if params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, len(g_conf.DATA_USED)*g_conf.ENCODER_INPUT_FRAMES_NUM*self.res_out_h*self.res_out_w, params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=params['TxEncoder']['d_model'], dropout=0.0, max_len=len(g_conf.DATA_USED)*g_conf.ENCODER_INPUT_FRAMES_NUM*self.res_out_h*self.res_out_w)

        join_dim = params['TxEncoder']['d_model']
        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=params['TxEncoder']['d_model'],
                                                   nhead=params['TxEncoder']['n_head'],
                                                   norm_first=params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(params['TxEncoder']['d_model']))

        self.action_output = FC(params={'neurons': [join_dim] +
                                            params['action_output']['fc']['neurons'] +
                                            [len(g_conf.TARGETS)],
                                 'dropouts': params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

    def forward(self, s, s_d, s_s, B):
        # 所有 tensor 均假定已经有 batch 维度，不再添加 unsqueeze
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        # B = s_s[0].shape[0]  # 应该是 1

        # ========== 图像堆叠 [B, S, cam, 3, H, W] ==========
        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        # print("→ stacked x shape:", x.shape)
        x = x.view(B*S*len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]
        # print("final x shape:", x.shape)  # 应为 [1*S*cam, 3, H, W]
        # print("final d shape:", d.shape)  # [1, 4]
        # print("final s shape:", s.shape)  # [1, 1]
        # 图像嵌入
        # torch.save(x, "x_crash.pt")
        # e_p, _ = self.encoder_embedding_perception(x)    # [B*S*cam, dim, h, w]
        features = []
        with torch.no_grad():  # 如果是 inference 模式
            for i in range(0, x.shape[0], 1):  # 每次处理一个样本（你也可以改成 batch_size=4、8 试试）
                x_batch = x[i:i + 1]
                e_p_batch, _ = self.encoder_embedding_perception(x_batch)  # [1, dim, h, w]
                features.append(e_p_batch)

        e_p = torch.cat(features, dim=0)  # [B*S*cam, dim, h, w]
        if B > 1:
            print('1')

        encoded_obs = e_p.view(B, S*len(g_conf.DATA_USED), self.res_out_dim, self.res_out_h*self.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.command(d).unsqueeze(1)     # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s

        if self.params['TxEncoder']['learnable_pe']:
            # 位置编码 positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer 编码器多头自注意力层
        in_memory, _ = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, len(TARGETS))

        return action_output         # (B, 1, 1), (B, 1, len(TARGETS))

    def forward_eval(self, s, s_d, s_s):

        # s = [[img.to(self.device) for img in frame] for frame in s]  # [S][cam][B, 3, H, W]
        # s_d = [x.to(self.device) for x in s_d]  # 每个都是 [B, dim]
        # s_s = [x.to(self.device) for x in s_s]  # 每个都是 [B, 1]
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(g_conf.DATA_USED), *g_conf.IMAGE_SHAPE)  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]
        # batch_size = 164
        # features = []
        # with torch.no_grad():  # 前向提取特征，不需要梯度
        #     for i in range(0, B, batch_size):
        #         x_batch = x[i:i + batch_size]
        #         e_p_batch, _ = self.encoder_embedding_perception(x_batch)  # shape [bs, dim, h, w]
        #         features.append(e_p_batch)

        # e_p = torch.cat(features, dim=0)  # [B*S*cam, dim, h, w]
        # 图像嵌入
        # e_p, resnet_inter = self.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
        features = []
        with torch.no_grad():  # 如果是 inference 模式
            for i in range(0, x.shape[0], 1):  # 每次处理一个样本
                x_batch = x[i:i + 1]
                e_p_batch, _ = self.encoder_embedding_perception(x_batch)  # [1, dim, h, w]
                features.append(e_p_batch)

        e_p = torch.cat(features, dim=0)  # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S * len(g_conf.DATA_USED), self.res_out_dim,  self.res_out_h * self.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.command(d).unsqueeze(1)  # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s   # 当前的观测=图片特侦+命令+速度 [B, S*cam*h*w, 512]

        if self.params['TxEncoder']['learnable_pe']:
            # 位置编码
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer 编码器多头自注意力层
        in_memory, attn_weights = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]
        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, len(TARGETS))

        return action_output, in_memory, attn_weights

    def generate_square_subsequent_mask(self, sz):
        r"""为序列生成一个方形掩码。掩码位置用 float('-inf') 填充。
            未屏蔽的位置用浮点数（0.0）填充。
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

