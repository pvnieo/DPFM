from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_net.layers import DiffusionNet
from utils import get_mask, nn_interpolate


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    torch.cuda.empty_cache()
    prob = torch.nn.functional.softmax(scores, dim=-1)
    torch.cuda.empty_cache()
    result = torch.einsum("bhnm,bdhm->bdhn", prob, value)
    torch.cuda.empty_cache()
    return result, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) for ll, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class CrossAttentionRefinementNet(nn.Module):
    def __init__(self, n_in=128, num_head=4, gnn_dim=512, overlap_feat_dim=32, n_layers=2, cross_sampling_ratio=0.15, attention_type="normal"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type == "normal":
            additional_dim = 0
            overlap_feat_dim = n_in
        elif attention_type == "double":
            additional_dim = overlap_feat_dim
        else:
            raise Exception("Attention type not recognized")

        self.n_in = n_in
        self.cross_sampling_ratio = cross_sampling_ratio
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(AttentionalPropagation(gnn_dim + additional_dim, num_head))
        self.layers = nn.ModuleList(self.layers)

        self.first_lin = nn.Linear(n_in, gnn_dim + additional_dim)
        self.last_lin = nn.Linear(gnn_dim + additional_dim, n_in + additional_dim)

        self.overlap_predictor = OverlapPredictorNet(overlap_feat_dim=overlap_feat_dim)

    def forward(self, coords0, coords1, features_x, features_y, batch):
        desc0, desc1 = self.first_lin(features_x).transpose(1, 2), self.first_lin(features_y).transpose(1, 2)

        for layer in self.layers:
            flat_coords0, flat_coords1 = coords0.squeeze(0).t(), coords1.squeeze(0).t()
            if self.cross_sampling_ratio == 1:
                desc0 = desc0 + layer(desc0, desc1)
                torch.cuda.empty_cache()
                desc1 = desc1 + layer(desc1, desc0)
                torch.cuda.empty_cache()
            else:
                n0, n1 = int(self.cross_sampling_ratio * flat_coords0.shape[1]), int(self.cross_sampling_ratio * flat_coords1.shape[1])
                idf0, idn0, dists0 = batch["shape1"]["sample_idx"]
                idf1, idn1, dists1 = batch["shape2"]["sample_idx"]
                idf0, idn0, dists0 = idf0[:n0], idn0, dists0
                idf1, idn1, dists1 = idf1[:n1], idn1, dists1
                sampled_desc0, sampled_desc1 = desc0[:, :, idf0], desc1[:, :, idf1]
                sampled_desc0 = sampled_desc0 + layer(sampled_desc0, sampled_desc1)
                torch.cuda.empty_cache()
                sampled_desc1 = sampled_desc1 + layer(sampled_desc1, sampled_desc0)
                torch.cuda.empty_cache()
                desc0 = nn_interpolate(sampled_desc0.transpose(1, 2), flat_coords0.t(), dists0, idn0, idf0).transpose(1, 2)
                desc1 = nn_interpolate(sampled_desc1.transpose(1, 2), flat_coords1.t(), dists1, idn1, idf1).transpose(1, 2)

        augmented_features_x = self.last_lin(desc0.transpose(1, 2))
        augmented_features_y = self.last_lin(desc1.transpose(1, 2))

        ref_feat_x, ref_feat_y = augmented_features_x[:, :, :self.n_in], augmented_features_y[:, :, :self.n_in]
        if self.attention_type == "normal":
            overlap_score_x, overlap_score_y = self.overlap_predictor(ref_feat_x, ref_feat_y)
        elif self.attention_type == "double":
            overlap_feat_x, overlap_feat_y = augmented_features_x[:, :, self.n_in:], augmented_features_y[:, :, self.n_in:]
            overlap_score_x, overlap_score_y = self.overlap_predictor(overlap_feat_x, overlap_feat_y)

        return ref_feat_x, ref_feat_y, overlap_score_x, overlap_score_y


class OverlapPredictorNet(nn.Module):
    def __init__(self, overlap_feat_dim=32):
        super().__init__()

        self.overlap_score_net = nn.Sequential(
            nn.Linear(overlap_feat_dim, overlap_feat_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(overlap_feat_dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, overlap_feat_x, overlap_feat_y):
        norm_feat_x = F.normalize(overlap_feat_x, p=2, dim=-1)
        norm_feat_y = F.normalize(overlap_feat_y, p=2, dim=-1)

        overlap_score_x = self.overlap_score_net(norm_feat_x).squeeze(2).squeeze(0)
        overlap_score_y = self.overlap_score_net(norm_feat_y).squeeze(2).squeeze(0)

        return overlap_score_x, overlap_score_y


class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C


class DPFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg):
        super().__init__()

        # feature extractor
        self.feature_extractor = DiffusionNet(
            C_in=cfg["fmap"]["C_in"],
            C_out=cfg["fmap"]["n_feat"],
            C_width=128,
            N_block=4,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
        )

        # cross attention refinement
        self.feat_refiner = CrossAttentionRefinementNet(n_in=cfg["fmap"]["n_feat"], num_head=cfg["attention"]["num_head"], gnn_dim=cfg["attention"]["gnn_dim"],
                                                        overlap_feat_dim=cfg["overlap"]["overlap_feat_dim"],
                                                        n_layers=cfg["attention"]["ref_n_layers"],
                                                        cross_sampling_ratio=cfg["attention"]["cross_sampling_ratio"],
                                                        attention_type=cfg["attention"]["attention_type"])

        # regularized fmap
        self.fmreg_net = RegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])
        self.n_fmap = cfg["fmap"]["n_fmap"]
        self.robust = cfg["fmap"]["robust"]

    def forward(self, batch):
        verts1, faces1, mass1, L1, evals1, evecs1, gradX1, gradY1 = (batch["shape1"]["xyz"], batch["shape1"]["faces"], batch["shape1"]["mass"],
                                                                     batch["shape1"]["L"], batch["shape1"]["evals"], batch["shape1"]["evecs"],
                                                                     batch["shape1"]["gradX"], batch["shape1"]["gradY"])
        verts2, faces2, mass2, L2, evals2, evecs2, gradX2, gradY2 = (batch["shape2"]["xyz"], batch["shape2"]["faces"], batch["shape2"]["mass"],
                                                                     batch["shape2"]["L"], batch["shape2"]["evals"], batch["shape2"]["evecs"],
                                                                     batch["shape2"]["gradX"], batch["shape2"]["gradY"])

        # set features to vertices
        features1, features2 = verts1, verts2

        feat1 = self.feature_extractor(features1, mass1, L=L1, evals=evals1, evecs=evecs1,
                                       gradX=gradX1, gradY=gradY1, faces=faces1).unsqueeze(0)
        feat2 = self.feature_extractor(features2, mass2, L=L2, evals=evals2, evecs=evecs2,
                                       gradX=gradX2, gradY=gradY2, faces=faces2).unsqueeze(0)

        # refine features
        ref_feat1, ref_feat2, overlap_score12, overlap_score21 = self.feat_refiner(verts1, verts2, feat1, feat2, batch)
        use_feat1, use_feat2 = (ref_feat1, ref_feat2) if self.robust else (feat1, feat2)
        # predict fmap
        evecs_trans1, evecs_trans2 = evecs1.t()[:self.n_fmap] @ torch.diag(mass1), evecs2.t()[:self.n_fmap] @ torch.diag(mass2)
        evals1, evals2 = evals1[:self.n_fmap], evals2[:self.n_fmap]
        C_pred = self.fmreg_net(use_feat1, use_feat2, evals1, evals2, evecs_trans1, evecs_trans2)

        return C_pred, overlap_score12, overlap_score21, use_feat1, use_feat2
