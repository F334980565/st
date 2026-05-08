from packaging import version
import torch
from torch import nn
import torch.nn.functional as F
import time

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = (
            torch.uint8
            if version.parse(torch.__version__) < version.parse("1.2.0")
            else torch.bool
        )

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[
            None, :, :
        ]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )
        return loss

class OA_NCE_Loss(nn.Module):
    def __init__(
        self, tau=0.07, tau_w=0.07, eps=0.1, tau_ot=0.1, max_iter=50, alpha=1.0
    ):
        super(OA_NCE_Loss, self).__init__()
        self.tau = tau
        self.tau_w = tau_w
        self.eps = eps
        self.tau_ot = tau_ot
        self.max_iter = max_iter
        self.alpha = alpha

    def sinkhorn_unbalanced(self, A, B):
        b, n, c = A.shape
        sim = torch.bmm(A, B.transpose(1, 2))
        cost = 1.0 - sim
        K = torch.exp(-cost / self.eps)
        u = torch.ones(b, n, 1, device=A.device) / n
        v = torch.ones(b, n, 1, device=A.device) / n
        fi = self.tau_ot / (self.tau_ot + self.eps)
        for _ in range(self.max_iter):
            u = (1.0 / (torch.bmm(K, v) + 1e-8)) ** fi
            v = (1.0 / (torch.bmm(K.transpose(1, 2), u) + 1e-8)) ** fi
        P = u * K * v.transpose(1, 2)
        return P

    def forward(self, feat_src, feat_gen, feat_tgt, src_mask=None):
        B, N, C = feat_src.shape
        feat_src = F.normalize(feat_src, dim=-1)
        feat_gen = F.normalize(feat_gen, dim=-1)
        feat_tgt = F.normalize(feat_tgt, dim=-1)
        P_x = self.sinkhorn_unbalanced(feat_src, feat_tgt)
        P_y = self.sinkhorn_unbalanced(feat_gen, feat_tgt)
        a_mat = torch.sqrt(P_x * P_y + 1e-10)
        a = a_mat.sum(dim=2)
        k = max(1, int(N * 0.2))
        topk_values, _ = torch.topk(a, k, dim=1)
        threshold_k = topk_values[:, -1].unsqueeze(1)
        robust_mask = (a >= threshold_k).detach().float()
        if src_mask is not None:
            robust_mask = robust_mask * src_mask
        a_for_w = a.clone()
        a_for_w[robust_mask == 0] = -1e9
        a_max, _ = torch.max(a_for_w, dim=1, keepdim=True)
        w = torch.exp((a_for_w - a_max.detach()) / self.tau_w)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        w = w.detach()
        logits_pos = torch.sum(feat_src * feat_gen, dim=-1) / self.tau
        logits_all = torch.bmm(feat_src, feat_gen.transpose(1, 2)) / self.tau
        logits_max, _ = torch.max(logits_all, dim=2, keepdim=True)
        exp_all = torch.exp(logits_all - logits_max.detach())
        log_sum_exp = logits_max.squeeze(-1) + torch.log(exp_all.sum(dim=2) + 1e-10)
        nce_term = (w * (log_sum_exp - logits_pos)).sum() / B
        reg_consistency = -self.alpha * (a * robust_mask).sum() / (B * k) * 10
        loss_total = nce_term + reg_consistency
        self.last_a_score = topk_values.mean().item()
        return loss_total

class AI_NCE_Loss(nn.Module):
    def __init__(self, tau=0.07):
        super(AI_NCE_Loss, self).__init__()
        self.tau = tau

    def forward(self, x, y_ref):
        B, N, C = y_ref.shape
        x = F.normalize(x, dim=-1)
        y_ref = F.normalize(y_ref, dim=-1)
        y_all = y_ref.reshape(-1, C)
        logits_all = torch.mm(x, y_all.t()) / self.tau
        logits_max, _ = torch.max(logits_all, dim=1, keepdim=True)
        exp_sum = torch.exp(logits_all - logits_max).sum(dim=1)
        log_sum_exp = torch.log(exp_sum + 1e-10) + logits_max.squeeze(1)
        logits_pos = torch.sum(x.unsqueeze(1) * y_ref, dim=-1) / self.tau
        log_sum_exp_expanded = log_sum_exp.unsqueeze(1).repeat(1, N)
        per_patch_loss = log_sum_exp_expanded - logits_pos
        loss = per_patch_loss.mean()
        return loss
