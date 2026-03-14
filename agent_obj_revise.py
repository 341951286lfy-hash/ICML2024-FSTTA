

import copy
import math
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks

from .agent_base import Seq2SeqAgent
from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad

# -----------------------------------------------------------------------------
# 指令规范化模块：
# 优先导入项目里正式实现的工具；如果暂时没有该文件，则退化为一个本地简化版，
# 这样至少 agent 文件本身是完整可读、可逐步集成的。
# -----------------------------------------------------------------------------
try:
    from utils.instruction_normalizer import InstructionNormalizer
except ImportError:
    import re

    class InstructionNormalizer:  # type: ignore
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

        def normalize(self, text):
            if text is None:
                return ""
            t = text.lower().strip()
            t = re.sub(r"[^\w\s]", " ", t)
            t = re.sub(r"\s+", " ", t)
            t = t.replace("move forward", "go forward")
            t = t.replace("walk forward", "go forward")
            t = t.replace("turn towards your right hand side", "turn right")
            t = t.replace("turn towards your left hand side", "turn left")
            return t.strip()

        def encode(self, text):
            if self.tokenizer is None:
                return text.split()
            return self.tokenizer.encode(text)


class GMapObjectNavAgent(Seq2SeqAgent):

    def _build_model(self):
        """构建主模型，并初始化测试时适应（TTA）所需的懒加载状态。

        注意：
        这里不直接创建 optimizer，因为很多 VLN 代码库会在 `_build_model`
        之后再加载 checkpoint。为了避免参数加载顺序冲突，真正的冻结参数、
        构建 Env/Scene Adapter 的优化器，放到 rollout 第一次执行时再做。
        """
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()

        # 用于 `make_equiv_action` 中把图动作映射回 simulator 视角索引
        self.scanvp_cands = {}

        # 新的 adapter-based TTA 流程所需标记和对象
        self._tta_initialized = False
        self.instr_norm = None
        self.optimizer_env = None
        self.optimizer_scene = None

    def _language_variable(self, obs):
        """把观测中的指令 token 序列整理成张量。"""
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool_)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor,
            'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        """提取预计算好的全景视觉特征，整理为模型输入格式。"""
        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens = [], []
        batch_cand_vpids, batch_objids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []

            # 候选方向视角
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)  # 1 表示可导航候选视角
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])

            # 非候选视角
            view_img_fts.extend([
                x[:self.args.image_feat_size] for k, x in enumerate(ob['feature'])
                if k not in used_viewidxs
            ])
            view_ang_fts.extend([
                x[self.args.image_feat_size:] for k, x in enumerate(ob['feature'])
                if k not in used_viewidxs
            ])
            nav_types.extend([0] * (36 - len(used_viewidxs)))  # 0 表示普通视角

            # 拼接视角特征与位置特征
            view_img_fts = np.stack(view_img_fts, 0)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # 目标物体特征
            obj_loc_fts = np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)
            nav_types.extend([2] * len(obj_loc_fts))  # 2 表示 object token

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))
            batch_loc_fts.append(torch.from_numpy(np.concatenate([view_loc_fts, obj_loc_fts], 0)))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_objids.append(ob['obj_ids'])
            batch_view_lens.append(len(view_img_fts))
            batch_obj_lens.append(len(ob['obj_img_fts']))

        # padding 到 batch 内最大长度
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()
        batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts,
            'obj_img_fts': batch_obj_img_fts,
            'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types,
            'view_lens': batch_view_lens,
            'obj_lens': batch_obj_lens,
            'cand_vpids': batch_cand_vpids,
            'obj_ids': batch_objids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        """构建 global graph map 分支所需输入。"""
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)

            batch_no_vp_left.append(len(unvisited_vpids) == 0)

            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'],
                gmap_vpids,
                obs[i]['heading'],
                obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for ii in range(1, len(gmap_vpids)):
                for jj in range(ii + 1, len(gmap_vpids)):
                    gmap_pair_dists[ii, jj] = gmap_pair_dists[jj, ii] = \
                        gmap.graph.distance(gmap_vpids[ii], gmap_vpids[jj])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids,
            'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids,
            'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists,
            'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, obj_lens, nav_types):
        """构建局部 vp 分支所需输入。"""
        batch_size = len(obs)

        # 在序列最前加入 [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'],
                cand_vpids[i],
                obs[i]['heading'],
                obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'],
                [gmap.start_vp],
                obs[i]['heading'],
                obs[i]['elevation']
            )

            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)
        vp_obj_masks = torch.cat([torch.zeros(batch_size, 1).bool().cuda(), nav_types == 2], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + obj_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_obj_masks': vp_obj_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """生成 teacher forcing 下的动作标签。"""
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] + \
                                   self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % scan)

        return torch.from_numpy(a).cuda()

    def _teacher_object(self, obs, ended, view_lens):
        """生成 object grounding 的监督标签。"""
        targets = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                targets[i] = self.args.ignoreid
            else:
                i_vp = ob['viewpoint']
                if i_vp not in ob['gt_end_vps']:
                    targets[i] = self.args.ignoreid
                else:
                    i_objids = ob['obj_ids']
                    targets[i] = self.args.ignoreid
                    for j, obj_id in enumerate(i_objids):
                        if str(obj_id) == str(ob['gt_obj_id']):
                            targets[i] = j + view_lens[i] + 1
                            break
        return torch.from_numpy(targets).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """把 panoramic action 转换为 simulator 可执行的等价动作。"""
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        """更新当前 scan-viewpoint 下候选 viewpoint 到 pointId 的映射。"""
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # ------------------------------------------------------------------
    # 新增：adapter-based test-time adaptation 相关辅助函数
    # ------------------------------------------------------------------
    def _get_instruction_tokenizer(self):
        """尽量从常见字段中找到 tokenizer。

        不同 VLN 工程暴露 tokenizer 的位置不一样，所以这里做多处尝试。
        找不到就返回 None，此时规范化只保存文本，不强制重新 tokenize。
        """
        candidates = [
            getattr(self, 'tok', None),
            getattr(self, 'tokenizer', None),
            getattr(self, 'instr_tokenizer', None),
            getattr(self.env, 'tok', None) if hasattr(self, 'env') else None,
            getattr(self.env, 'tokenizer', None) if hasattr(self, 'env') else None,
        ]
        for tok in candidates:
            if tok is not None:
                return tok
        return None

    def _init_tta_components_if_needed(self):
        """只在第一次 rollout 时初始化 TTA 所需组件。

        这是原 FSTTA fast/slow 逻辑的替代核心：
        - 冻结 VLN backbone
        - 只训练 env_adapter / scene_adapter_g / scene_adapter_v
        - 创建相应 optimizer
        - 初始化 instruction normalizer
        """
        if self._tta_initialized:
            return

        # 从 args 读取超参数；若尚未在 config 中补齐，则给一个安全默认值
        self.env_lr = float(getattr(self.args, 'env_lr', 1e-4))
        self.scene_lr = float(getattr(self.args, 'scene_lr', 1e-4))
        self.env_k = int(getattr(self.args, 'env_k', 2))
        self.env_lambda_ent = float(getattr(self.args, 'env_lambda_ent', 0.1))
        self.beta_fail = float(getattr(self.args, 'scene_beta_fail', 0.3))
        self.use_step_weight = bool(getattr(self.args, 'scene_use_step_weight', True))
        self.min_step_weight = float(getattr(self.args, 'scene_min_step_weight', 0.2))
        self.enable_scene_sgr = bool(getattr(self.args, 'enable_scene_sgr', True))
        self.scene_reverse_prob = float(getattr(self.args, 'scene_reverse_prob', 0.1))
        self.scene_reverse_alpha = float(getattr(self.args, 'scene_reverse_alpha', -0.5))
        self.enable_scene_update = bool(getattr(self.args, 'enable_scene_update', True))
        self.feature_noise_std = float(getattr(self.args, 'feature_noise_std', 0.01))
        self.feature_drop_prob = float(getattr(self.args, 'feature_drop_prob', 0.05))

        # 文档要求的 adapter 必须已经在 vilmodel.py 中注册好
        required_modules = ['env_adapter', 'scene_adapter_g', 'scene_adapter_v']
        for module_name in required_modules:
            if not hasattr(self.vln_bert, module_name):
                raise AttributeError(
                    f"`self.vln_bert.{module_name}` is missing. "
                    "Please first apply the required `vilmodel.py` changes and "
                    "create the adapter modules mentioned in 在FSTTA上完善代码.md."
                )

        # 先冻结整个 backbone
        for p in self.vln_bert.parameters():
            p.requires_grad = False

        # 再只放开 adapter
        for p in self.vln_bert.env_adapter.parameters():
            p.requires_grad = True
        for p in self.vln_bert.scene_adapter_g.parameters():
            p.requires_grad = True
        for p in self.vln_bert.scene_adapter_v.parameters():
            p.requires_grad = True

        self.optimizer_env = torch.optim.Adam(
            self.vln_bert.env_adapter.parameters(),
            lr=self.env_lr
        )
        self.optimizer_scene = torch.optim.Adam(
            list(self.vln_bert.scene_adapter_g.parameters()) +
            list(self.vln_bert.scene_adapter_v.parameters()),
            lr=self.scene_lr,
        )

        self.instr_norm = InstructionNormalizer(tokenizer=self._get_instruction_tokenizer())
        self._tta_initialized = True

    def _set_env_adapter_trainable(self, enabled: bool):
        """控制 Env Adapter 是否允许梯度更新。"""
        for p in self.vln_bert.env_adapter.parameters():
            p.requires_grad = enabled

    def _set_scene_adapters_trainable(self, enabled: bool):
        """控制 Scene Adapter 是否允许梯度更新。"""
        for p in self.vln_bert.scene_adapter_g.parameters():
            p.requires_grad = enabled
        for p in self.vln_bert.scene_adapter_v.parameters():
            p.requires_grad = enabled

    def _normalize_obs_instructions(self, obs):
        """对观测中的文本指令做规则化预处理，并在可能时重新编码。"""
        if self.instr_norm is None:
            return obs

        tokenizer = getattr(self.instr_norm, 'tokenizer', None)
        for ob in obs:
            raw_text = ob.get('instruction', ob.get('instr', ''))
            norm_text = self.instr_norm.normalize(raw_text)
            ob['norm_instruction'] = norm_text

            if tokenizer is not None:
                encoded = self.instr_norm.encode(norm_text)
                if isinstance(encoded, dict):
                    encoded = encoded.get('input_ids', encoded)
                if torch.is_tensor(encoded):
                    encoded = encoded.detach().cpu().tolist()
                ob['instr_encoding'] = list(encoded)
            else:
                ob['instr_encoding'] = list(ob.get('instr_encoding', []))
        return obs

    def _augment_feature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """对预提取视觉特征做轻量增强。

        这里没有原始图像，所以采用 feature-space 的噪声扰动 + dropout，
        作为多视图一致性训练中的简化增强方式。
        """
        if x.numel() == 0:
            return x

        out = x.clone()
        if self.feature_noise_std > 0:
            out = out + torch.randn_like(out) * self.feature_noise_std
        if self.feature_drop_prob > 0:
            keep = (torch.rand_like(out) > self.feature_drop_prob).float()
            out = out * keep
        return out

    def _panorama_feature_variable_with_aug(self, obs, aug_id=1):
        """生成带增强版本的 panorama 输入。

        注意：
        结构必须和 `_panorama_feature_variable` 完全一致，只改 feature tensor，
        不改 metadata。
        """
        pano_inputs = self._panorama_feature_variable(obs)
        pano_inputs['view_img_fts'] = self._augment_feature_tensor(pano_inputs['view_img_fts'])
        pano_inputs['obj_img_fts'] = self._augment_feature_tensor(pano_inputs['obj_img_fts'])
        return pano_inputs

    def _build_k_pano_inputs(self, obs):
        """构建 1 条 clean route + K-1 条增强 route。"""
        pano_inputs_list = [self._panorama_feature_variable(obs)]
        for aug_id in range(1, self.env_k):
            pano_inputs_list.append(self._panorama_feature_variable_with_aug(obs, aug_id=aug_id))
        return pano_inputs_list

    def _unpack_panorama_output(self, pano_out):
        """兼容不同工程中 panorama 输出的 tuple / dict 两种格式。"""
        if isinstance(pano_out, tuple):
            pano_embeds, pano_masks = pano_out
        elif isinstance(pano_out, dict):
            pano_embeds = pano_out['pano_embeds']
            pano_masks = pano_out['pano_masks']
        else:
            raise TypeError(f'Unsupported panorama output type: {type(pano_out)}')
        return pano_embeds, pano_masks

    def _update_gmap_with_pano(self, obs, gmaps, pano_inputs, pano_embeds, pano_masks, ended):
        """仅使用 clean route 更新 persistent graph memory。

        文档要求：图记忆/历史节点嵌入只能由主路由更新，增强路由只用于
        Env Adapter 的一致性学习，不能污染图状态。
        """
        avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                          torch.sum(pano_masks, 1, keepdim=True)

        for i, gmap in enumerate(gmaps):
            if ended[i]:
                continue
            i_vp = obs[i]['viewpoint']
            gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
            for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                if not gmap.graph.visited(i_cand_vp):
                    gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

    def _build_nav_inputs(self, obs, gmaps, pano_embeds, pano_inputs, txt_embeds, txt_masks):
        """组装 navigation 阶段需要的完整输入字典。"""
        nav_inputs = self._nav_gmap_variable(obs, gmaps)
        nav_inputs.update(
            self._nav_vp_variable(
                obs,
                gmaps,
                pano_embeds,
                pano_inputs['cand_vpids'],
                pano_inputs['view_lens'],
                pano_inputs['obj_lens'],
                pano_inputs['nav_types'],
            )
        )
        nav_inputs.update({
            'txt_embeds': txt_embeds,
            'txt_masks': txt_masks,
        })
        return nav_inputs

    def _select_nav_logits_and_vpids(self, nav_outs, nav_inputs):
        """根据 fusion 模式选择实际用于动作决策的 logits 和候选 vpids。"""
        if self.args.fusion == 'local':
            nav_logits = nav_outs['local_logits']
            nav_vpids = nav_inputs['vp_cand_vpids']
        elif self.args.fusion == 'global':
            nav_logits = nav_outs['global_logits']
            nav_vpids = nav_inputs['gmap_vpids']
        else:
            nav_logits = nav_outs['fused_logits']
            nav_vpids = nav_inputs['gmap_vpids']
        return nav_logits, nav_vpids

    def _select_env_logits(self, nav_outs, nav_inputs):
        """Env loss 优先用 fused logits，没有的话退化到当前动作 logits。"""
        if 'fused_logits' in nav_outs:
            return nav_outs['fused_logits']
        nav_logits, _ = self._select_nav_logits_and_vpids(nav_outs, nav_inputs)
        return nav_logits

    def _compute_env_loss(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """计算 Env Adapter 的测试时损失。

        思路：
        - clean route 作为 anchor 分布
        - augmented routes 与 clean route 做 KL 风格一致性约束
        - 同时在 clean route 上做熵最小化，鼓励更确定的决策
        """
        p0 = torch.softmax(logits_list[0], dim=-1)
        log_p0 = torch.log_softmax(logits_list[0], dim=-1)

        loss_cons = 0.0
        for k in range(1, len(logits_list)):
            log_pk = torch.log_softmax(logits_list[k], dim=-1)
            loss_cons = loss_cons + torch.sum(p0 * (log_p0 - log_pk), dim=-1).mean()

        loss_cons = loss_cons / max(1, len(logits_list) - 1)
        loss_ent = -(p0 * log_p0).sum(dim=-1).mean()
        return loss_cons + self.env_lambda_ent * loss_ent

    def _detach_for_scene_cache(self, value):
        """把缓存内容递归 detach，供 episode 末重新前向计算 scene loss。

        这样做的原因：
        Env Adapter 会在每一步在线更新，如果直接保留旧计算图，到 episode
        末再反传会造成 autograd 图失效或版本冲突。
        """
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: self._detach_for_scene_cache(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._detach_for_scene_cache(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._detach_for_scene_cache(v) for v in value)
        return copy.deepcopy(value)

    def _apply_sgr_to_scene_adapter(self, p_rev=0.1, alpha=-0.5):
        """对 Scene Adapter 梯度施加随机梯度反转（SGR）。

        用于失败 episode 的反向调节：部分参数梯度翻转，其余参数做归一化缩放。
        """
        params = list(self.vln_bert.scene_adapter_g.parameters()) + \
                 list(self.vln_bert.scene_adapter_v.parameters())
        denom = alpha * p_rev + (1.0 - p_rev)
        if abs(denom) < 1e-8:
            return

        for p in params:
            if p.grad is None:
                continue
            mask = (torch.rand_like(p.grad) < p_rev).float()
            p.grad = mask * (alpha * p.grad) + (1.0 - mask) * (p.grad / denom)

    def _compute_scene_loss_from_cache(
        self,
        scene_step_cache,
        success_flags,
        beta_fail=0.3,
        use_step_weight=True,
        min_step_weight=0.2,
    ):
        """基于缓存的 clean-route 导航状态，重新前向并计算 trajectory-level NLL。

        规则：
        - 成功轨迹：提升已执行动作概率，即正常最小化 NLL
        - 失败轨迹：弱化地抑制其动作序列，即用负权重项
        """
        if len(scene_step_cache) == 0:
            return None

        per_traj_losses = [[] for _ in range(len(success_flags))]

        total_steps_per_sample = [0 for _ in range(len(success_flags))]
        for cache in scene_step_cache:
            active_mask = cache['active_mask']
            nav_outs = self.vln_bert('navigation', cache['nav_inputs'])
            logits, _ = self._select_nav_logits_and_vpids(nav_outs, cache['nav_inputs'])
            log_probs = torch.log_softmax(logits, dim=-1)

            for i in range(len(success_flags)):
                if not active_mask[i]:
                    continue
                act_i = int(cache['actions'][i].item())
                nll_i = -log_probs[i, act_i]
                per_traj_losses[i].append(nll_i)
                total_steps_per_sample[i] += 1

        losses = []
        for i, step_losses in enumerate(per_traj_losses):
            if len(step_losses) == 0:
                continue

            weighted_losses = []
            T = len(step_losses)
            for t, nll_t in enumerate(step_losses):
                if use_step_weight:
                    if T == 1:
                        w_t = 1.0
                    else:
                        w_t = min_step_weight + (1.0 - min_step_weight) * (t / (T - 1))
                else:
                    w_t = 1.0
                weighted_losses.append(w_t * nll_t)

            traj_nll = torch.stack(weighted_losses).mean()
            if success_flags[i]:
                losses.append(traj_nll)
            else:
                losses.append(-beta_fail * traj_nll)

        if len(losses) == 0:
            return None
        return torch.stack(losses).mean()

    def rollout(self, TTA_lr_r, sar_margin_e0_r, output_dir_r, train_ml=None, train_rl=False, reset=True):
        """执行一次完整导航 rollout，并嵌入 adapter-based TTA 逻辑。

        相比原 FSTTA 版本，这里移除了 fast/slow 更新，改为：
        1. 指令规范化
        2. 每步 Env Adapter 更新
        3. episode 末 Scene Adapter 更新
        """
        del TTA_lr_r, sar_margin_e0_r, output_dir_r  # 新方案中不再使用

        # 首次执行时初始化 adapter / optimizer / tokenizer / freeze 状态
        self._init_tta_components_if_needed()

        if reset:
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # episode 开始时先做一次指令规范化
        obs = self._normalize_obs_instructions(obs)
        self._update_scanvp_cands(obs)

        batch_size = len(obs)

        # 初始化 graph memory
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # 记录轨迹
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'predObjId': None,
            'details': {},
        } for ob in obs]

        # 编码规范化后的指令
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)

        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # 保留原 agent 的监督损失接口
        ml_loss = 0.0
        og_loss = 0.0
        entropys = []

        # 缓存 clean-route 的 navigation 输入，episode 末做 scene-level 更新
        scene_step_cache = []

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # --------------------------------------------------------------
            # Step 1: 构造一条 clean route 和 K-1 条增强 route
            # --------------------------------------------------------------
            pano_inputs_list = self._build_k_pano_inputs(obs)
            nav_out_list = []
            nav_inputs_list = []

            # Env Adapter 更新阶段，冻结 Scene Adapter
            self._set_env_adapter_trainable(True)
            self._set_scene_adapters_trainable(False)
            self.optimizer_env.zero_grad(set_to_none=True)
            if self.optimizer_scene is not None:
                self.optimizer_scene.zero_grad(set_to_none=True)

            for k, pano_inputs_k in enumerate(pano_inputs_list):
                pano_out_k = self.vln_bert('panorama', pano_inputs_k)
                pano_embeds_k, pano_masks_k = self._unpack_panorama_output(pano_out_k)

                # 只有 clean route 能更新 persistent graph
                if k == 0:
                    self._update_gmap_with_pano(
                        obs, gmaps, pano_inputs_k, pano_embeds_k, pano_masks_k, ended
                    )

                nav_inputs_k = self._build_nav_inputs(
                    obs,
                    gmaps,
                    pano_embeds_k,
                    pano_inputs_k,
                    txt_embeds,
                    language_inputs['txt_masks'],
                )
                nav_out_k = self.vln_bert('navigation', nav_inputs_k)
                nav_inputs_list.append(nav_inputs_k)
                nav_out_list.append(nav_out_k)

            # --------------------------------------------------------------
            # Step 2: 用 clean-vs-aug consistency 更新 Env Adapter
            # --------------------------------------------------------------
            env_logits_list = [
                self._select_env_logits(out, nav_inputs_list[idx])
                for idx, out in enumerate(nav_out_list)
            ]
            env_loss = self._compute_env_loss(env_logits_list)
            env_loss.backward()
            self.optimizer_env.step()
            self.optimizer_env.zero_grad(set_to_none=True)

            # 为后续 scene-level update 恢复 Scene Adapter 梯度
            self._set_scene_adapters_trainable(True)

            # clean route 决定实际动作与 scene cache
            nav_outs = nav_out_list[0]
            nav_inputs = nav_inputs_list[0]
            pano_inputs = pano_inputs_list[0]
            nav_logits, nav_vpids = self._select_nav_logits_and_vpids(nav_outs, nav_inputs)
            nav_probs = torch.softmax(nav_logits, dim=1)
            obj_logits = nav_outs['obj_logits']

            # 缓存 clean-route 的 navigation 输入
            scene_step_cache.append({
                'nav_inputs': self._detach_for_scene_cache(nav_inputs),
                'actions': torch.zeros(batch_size, dtype=torch.long, device=nav_logits.device),
                'active_mask': (~ended).copy(),
            })

            # 更新图中 stop/object 相关信息
            for i, gmap in enumerate(gmaps):
                if ended[i]:
                    continue
                i_vp = obs[i]['viewpoint']
                i_objids = obs[i]['obj_ids']
                i_obj_logits = obj_logits[i, pano_inputs['view_lens'][i] + 1:]
                gmap.node_stop_scores[i_vp] = {
                    'stop': nav_probs[i, 0].data.item(),
                    'og': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                    'og_details': {
                        'objids': i_objids,
                        'logits': i_obj_logits[:len(i_objids)],
                    },
                }

            # 保留 imitation learning / object grounding 训练接口
            if train_ml is not None:
                nav_targets = self._teacher_action(
                    obs,
                    nav_vpids,
                    ended,
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                )
                ml_loss += self.criterion(nav_logits, nav_targets)

                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)
                    local_nav_targets = self._teacher_action(
                        obs,
                        nav_inputs['vp_cand_vpids'],
                        ended,
                        visited_masks=None,
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)

                obj_targets = self._teacher_object(obs, ended, pano_inputs['view_lens'])
                og_loss += self.criterion(obj_logits, obj_targets)
            else:
                nav_targets = None

            # --------------------------------------------------------------
            # Step 3: 只基于 clean route 决定实际动作
            # --------------------------------------------------------------
            if self.feedback == 'teacher':
                a_t = nav_targets
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())
                entropys.append(c.entropy())
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size,) > self.args.expl_max_ratio
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (
                        nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()
                    ).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # 记录当前步实际执行的动作，供 episode 末 scene loss 使用
            scene_step_cache[-1]['actions'] = a_t.detach().clone()

            if self.feedback in ['teacher', 'sample']:
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            # 在 simulator 中执行动作
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)

            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k

                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

                    traj[i]['pred_objid'] = stop_score['og']

                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist(),
                            }

            # 刷新 observation 与图结构
            obs = self.env._get_obs()
            obs = self._normalize_obs_instructions(obs)
            self._update_scanvp_cands(obs)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))
            if ended.all():
                break

        # ------------------------------------------------------------------
        # Episode 结束后，执行 Scene Adapter 更新
        # ------------------------------------------------------------------
        if self.enable_scene_update and len(scene_step_cache) > 0:
            # 简单二值成功标记：最终预测 viewpoint 是否等于 gt 终点
            success_flags = []
            for i in range(batch_size):
                pred_vp = traj[i]['path'][-1][-1]
                gt_vp = obs[i]['gt_path'][-1]
                success_flags.append(pred_vp == gt_vp)

            # Scene update 不应反传到 Env Adapter
            self._set_env_adapter_trainable(False)
            self._set_scene_adapters_trainable(True)
            self.optimizer_scene.zero_grad(set_to_none=True)

            scene_loss = self._compute_scene_loss_from_cache(
                scene_step_cache,
                success_flags,
                beta_fail=self.beta_fail,
                use_step_weight=self.use_step_weight,
                min_step_weight=self.min_step_weight,
            )

            if scene_loss is not None:
                scene_loss.backward()

                # 若 episode 中存在失败轨迹，则对 Scene Adapter 梯度施加 SGR
                if self.enable_scene_sgr and (not all(success_flags)):
                    self._apply_sgr_to_scene_adapter(
                        p_rev=self.scene_reverse_prob,
                        alpha=self.scene_reverse_alpha,
                    )

                self.optimizer_scene.step()
                self.optimizer_scene.zero_grad(set_to_none=True)

            # 恢复 Env Adapter 可训练，供下一个 rollout 使用
            self._set_env_adapter_trainable(True)

        # 保留原始监督损失日志与统计
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            og_loss = og_loss * train_ml / batch_size
            self.loss += ml_loss
            self.loss += og_loss
            self.logs['IL_loss'].append(ml_loss.item())
            self.logs['OG_loss'].append(og_loss.item())

        return traj