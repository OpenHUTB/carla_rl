import torch
import torch.nn as nn
import importlib
import numpy as np

from configs import g_conf
from CILv2.models.building_blocks import FC
from CILv2.models.building_blocks.PositionalEncoding import PositionalEncoding
from CILv2.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from CILv2.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer


class CIL_multiview(nn.Module):
    def __init__(self, params):
        super(CIL_multiview, self).__init__()
        self.params = params
        self.projection = nn.Linear(256, 512)  # 将 256 维 -> 512 维
        resnet_module = importlib.import_module('CILv2.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf.IMAGENET_PRE_TRAINED,
                                                          layer_id = params['encoder_embedding']['perception']['res']['layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf.BATCH_SIZE] + g_conf.IMAGE_SHAPE)[params['encoder_embedding']['perception']['res']['layer_id']]

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

    def forward(self, s, s_d, s_s):
        # 输入统一处理（同时兼容Tensor和numpy输入）
        if isinstance(s, list):
            if isinstance(s[0], list):
                # 处理双层列表输入 [[img1, img2,...]]
                frame_list = []
                for img in s[0]:
                    if isinstance(img, np.ndarray):
                        # numpy数组处理
                        frame_list.append(torch.from_numpy(img).float().permute(2, 0, 1))
                    elif isinstance(img, torch.Tensor):
                        # 已经是张量的情况
                        if img.dim() == 3 and img.size(0) == 3:  # 已经是CHW格式
                            frame_list.append(img.float())
                        else:
                            frame_list.append(img.float().permute(2, 0, 1))
                    else:
                        raise TypeError(f"不支持的图像类型: {type(img)}")
                s = torch.stack(frame_list, dim=0)  # [N,C,H,W]
            else:
                # 处理单层列表输入 [img]
                img = s[0]
                if isinstance(img, np.ndarray):
                    s = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
                elif isinstance(img, torch.Tensor):
                    s = img.float().permute(2, 0, 1).unsqueeze(0) if img.dim() == 3 else img.unsqueeze(0)
        elif isinstance(s, torch.Tensor):
            # 已经是张量输入
            if s.dim() == 3:  # [C,H,W]
                s = s.unsqueeze(0)  # [1,C,H,W]
            elif s.dim() == 4:  # [B,C,H,W]
                pass
            else:
                raise ValueError(f"非法张量维度: {s.dim()}")
        else:
            raise TypeError(f"不支持的输入类型: {type(s)}")

        # 添加序列和相机维度 (S=1, cam=1)
        x = s.unsqueeze(1).unsqueeze(1)  # [B,1,1,C,H,W]

        # 获取批量大小
        B = x.size(0)

        # 重塑图像数据
        x = x.reshape(B * 1 * 1, *g_conf.IMAGE_SHAPE)  # [B,C,H,W]

        # 图像特征提取
        e_p, _ = self.encoder_embedding_perception(x)  # [B,512,10,20]

        # 处理速度和命令输入
        d = s_d[-1].view(B, -1)  # [4] -> [B, 4]
        s = s_s[-1].view(B, -1)  # [1] -> [B, 1]

        # 特征重塑
        encoded_obs = e_p.view(B, 1 * 1, self.res_out_dim, -1)  # [B,1,512,200]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B,200,512]
        # 投影层（如果存在）
        if hasattr(self, 'projection'):
            encoded_obs = self.projection(encoded_obs)

        # 命令和速度嵌入
        e_d = self.command(d).unsqueeze(1)  # [B,1,512]
        e_s = self.speed(s).unsqueeze(1)  # [B,1,512]

        # 特征融合
        encoded_obs = encoded_obs + e_d + e_s

        # 位置编码
        if hasattr(self, 'positional_encoding'):
            if self.params['TxEncoder']['learnable_pe']:
                pe = encoded_obs + self.positional_encoding
            else:
                pe = self.positional_encoding(encoded_obs)
        else:
            pe = encoded_obs

        # Transformer编码
        if hasattr(self, 'tx_encoder'):
            in_memory, _ = self.tx_encoder(pe)  # [B,200,512]
            in_memory = torch.mean(in_memory, dim=1)  # [B,512]
        else:
            in_memory = torch.mean(encoded_obs, dim=1)  # [B,512]

        # 动作输出
        action_output = self.action_output(in_memory)  # [B,action_dim]

        return action_output.unsqueeze(1)  # [B,1,action_dim]

    def foward_eval(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # 图像嵌入
        e_p, resnet_inter = self.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
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

        return action_output, resnet_inter, attn_weights


    def generate_square_subsequent_mask(self, sz):
        r"""为序列生成一个方形掩码。掩码位置用 float('-inf') 填充。
            未屏蔽的位置用浮点数（0.0）填充。
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

