import os
from collections import OrderedDict
import torch
from einops import rearrange
import util.util as util
from . import networks
from .base_model import BaseModel
from .patchnce import AI_NCE_Loss, OA_NCE_Loss, PatchNCELoss

class HISTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            "--cut_mode", type=str, default="cut", choices=["cut", "FastCUT", "fastcut"]
        )
        parser.add_argument(
            "--lambda_GAN",
            type=float,
            default=1.0,
            help="weight for GAN loss: GAN(G(X))",
        )
        parser.add_argument(
            "--lambda_NCE",
            type=float,
            default=1.0,
            help="weight for NCE loss: NCE(G(X), X)",
        )
        parser.add_argument(
            "--nce_idt",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="use NCE loss for identity mapping: NCE(G(Y), Y))",
        )
        parser.add_argument(
            "--nce_blocks",
            type=str,
            default="0,1,2,3,4,5",
            help="compute NCE loss on which blocks",
        )
        parser.add_argument(
            "--nce_includes_all_negatives_from_minibatch",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.",
        )
        parser.add_argument(
            "--netF",
            type=str,
            default="mlp_sample",
            choices=["mlp_sample"],
            help="how to downsample the feature map",
        )
        parser.add_argument("--netF_nc", type=int, default=256)
        parser.add_argument("--netF_seq_nc", type=int, default=512)
        parser.add_argument(
            "--nce_T", type=float, default=0.07, help="temperature for NCE loss"
        )
        parser.add_argument(
            "--num_patches", type=int, default=256, help="number of patches per layer"
        )
        parser.add_argument(
            "--flip_equivariance",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT",
        )
        parser.add_argument("--n_blocks", type=int, default=6)
        parser.add_argument("--res_out_i", type=int, default=1)
        parser.add_argument("--res_in_j", type=int, default=3)
        parser.add_argument("--mod_features", type=int, default=512)
        parser.add_argument(
            "--style_up", type=bool, default=False, help="use upsample style conv"
        )
        parser.add_argument("--n_local_blocks", type=int, default=2)
        parser.add_argument("--n_global_blocks", type=int, default=2)
        parser.add_argument("--instance_size", type=int, default=256)
        parser.add_argument("--local_encode_mode", type=str, choices=["trans", "conv"])
        parser.add_argument("--global_encode_mode", type=str, choices=["trans"])
        parser.add_argument(
            "--lambda_OA", type=float, default=0.0, help="weight for encode mode"
        )
        parser.add_argument(
            "--lambda_AI", type=float, default=0.0, help="weight for encode mode"
        )
        parser.add_argument(
            "--lambda_idt", type=float, default=0.0, help="weight for encode mode"
        )
        parser.add_argument(
            "--lambda_idt_B", type=float, default=0.0, help="weight for encode mode"
        )
        parser.add_argument(
            "--lambda_L1",
            type=float,
            default=0.0,
            help="weight for paired pixel reconstruction",
        )
        parser.set_defaults(pool_size=0)
        opt, _ = parser.parse_known_args()
        if opt.cut_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.cut_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False,
                lambda_NCE=10.0,
                flip_equivariance=True,
                n_epochs=150,
                n_epochs_decay=50,
            )
        else:
            raise ValueError(opt.cut_mode)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.loss_names = [
                "G_GAN",
                "D_real",
                "D_fake",
                "G",
                "NCE",
                "OA",
                "AI",
                "idt",
                "idt_B",
                "L1",
            ]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.nce_blocks = [int(x) for x in opt.nce_blocks.split(",")]
        self.instance_size = opt.instance_size
        if opt.nce_idt and self.isTrain:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["idt_B"]
        if self.isTrain:
            self.model_names = ["G", "F", "D"]
        else:
            self.model_names = ["G"]
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )
        self.netF = networks.define_F(
            opt.input_nc,
            opt.netF,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            self.gpu_ids,
            opt,
        )
        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            for block_id in self.nce_blocks:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionOA = OA_NCE_Loss(
                tau=0.07, tau_w=0.05, eps=0.1, tau_ot=0.1, max_iter=50, alpha=1.0
            )
            self.criterionAI = AI_NCE_Loss(tau=0.05)
            self.optimizer_G = torch.optim.Adam(
                [p for p in self.netG.parameters() if p.requires_grad],
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_k(self, k):
        self.k = k

    def data_dependent_initialize(self, data, style_start=100):
        B, C, H, W = data["A"].shape
        self.batch_size = B
        self.k = H // self.instance_size
        self.set_input(data)
        with torch.no_grad():
            self.netG._set_model(style_start, self.k)
        self.forward()
        if self.opt.isTrain:
            if self.opt.lambda_NCE > 0.0:
                self.netF.create_mlp_l(self.feats_l_realA)
            self.netF.create_mlp_i(
                [self.feats_i_realA, self.refined_feats_i_realA, self.feats_i_realB]
            )
            self.netF.create_mlp_m([self.feats_m_realA, self.feats_m_realB])
            self.netF.create_mlp_oa([self.feats_m_realA, self.feats_m_realB])
            self.optimizer_F = torch.optim.Adam(
                self.netF.parameters(),
                lr=self.opt.lr,
                betas=(self.opt.beta1, self.opt.beta2),
            )
            self.optimizers.append(self.optimizer_F)
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.clip_gradients(self.netD)
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.clip_gradients(self.netG, self.netF)
        self.optimizer_G.step()
        self.optimizer_F.step()

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        self.real_A = rearrange(
            self.real_A, "b c (p1 h) (p2 w) -> (b p1 p2) c h w", p1=self.k, p2=self.k
        )
        self.real_B = rearrange(
            self.real_B, "b c (p1 h) (p2 w) -> (b p1 p2) c h w", p1=self.k, p2=self.k
        )

    def forward(self):
        B = self.batch_size
        B_split = self.real_A.shape[0]
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        feats_l = []
        x, feats_l_ = self.netG.forward_encode(self.real, blocks=self.nce_blocks)
        feats_l += feats_l_
        x, feats_l_, feats_m, feats_i, refined_feats_i, feat_g, style = (
            self.netG.forward_bottle(x, blocks=self.nce_blocks, k=self.k)
        )
        feats_l += feats_l_
        x, feats_l_ = self.netG.forward_decode(x, style, blocks=self.nce_blocks)
        feats_l += feats_l_
        self.fake_B = x[:B_split]
        self.idt_B = x[B_split:]
        if self.isTrain:
            self.feats_l_realA = [feat[:B] for feat in feats_l]
            self.feats_l_realB = [feat[B:] for feat in feats_l]
            self.feats_m_realA = feats_m[:B_split]
            self.feats_m_realB = feats_m[B_split:]
            self.feats_i_realA = feats_i[:B_split]
            self.feats_i_realB = feats_i[B_split:]
            self.refined_feats_i_realA = refined_feats_i[:B_split]
            self.refined_feats_i_realB = refined_feats_i[B_split:]
            self.feats_g_realA = feat_g[:B]
            self.feats_g_realB = feat_g[B:]
            feats_l = []
            self.fake = torch.cat((self.fake_B, self.idt_B), dim=0)
            x, feats_l_ = self.netG.forward_encode(x, blocks=self.nce_blocks)
            feats_l += feats_l_
            x, feats_l_, feats_m, feats_i, refined_feat_i, feat_g, style = (
                self.netG.forward_bottle(x, blocks=self.nce_blocks, k=self.k)
            )
            feats_l += feats_l_
            _, feats_l_ = self.netG.forward_decode(x, style, blocks=self.nce_blocks)
            feats_l += feats_l_
            self.feats_l_fakeB = [feat[:B] for feat in feats_l]
            self.feats_l_idtB = [feat[B:] for feat in feats_l]
            self.feats_m_fakeB = feats_m[:B_split]
            self.feats_m_idtB = feats_m[B_split:]
            self.feats_i_fakeB = feats_i[:B_split]
            self.feats_i_idtB = feats_i[B_split:]
            self.feats_g_fakeB = feat_g[:B]
            self.feats_g_idtB = feat_g[B:]

    def compute_D_loss(self):
        if self.opt.lambda_GAN > 0.0:
            fake = self.fake_B.detach()
            real = self.real_B
            idt = self.idt_B
            if self.opt.netD == "cdan":
                fake_condition = self.feats_i_fakeB.detach()
                real_condition = self.feats_i_realB.detach()
                pred_fake = self.netD(fake, fake_condition)
                pred_real = self.netD(real, real_condition)
                indices = torch.randperm(real.size(0)).to(real.device)
                real_condition_shuffled = real_condition[indices]
                pred_mismatch = self.netD(real, real_condition_shuffled)
                self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
                self.loss_D_real = self.criterionGAN(pred_real, True).mean()
                self.loss_D_mismatch = self.criterionGAN(pred_mismatch, True).mean()
                self.loss_D = (
                    self.loss_D_fake + self.loss_D_real + self.loss_D_mismatch
                ) / 3.0
            else:
                pred_fake = self.netD(fake)
                pred_real = self.netD(real)
                self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
                self.loss_D_real = self.criterionGAN(pred_real, True).mean()
                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        else:
            self.loss_D = 0.0
        return self.loss_D

    def compute_G_loss(self):
        fake = self.fake_B
        idt = self.idt_B
        if self.opt.lambda_GAN > 0.0:
            if self.opt.netD == "cdan":
                pred_fake = self.netD(fake, self.feats_i_fakeB.detach())
                self.loss_G_GAN = (
                    self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
                )
            else:
                pred_fake = self.netD(fake)
                self.loss_G_GAN = (
                    self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
                )
        else:
            self.loss_G_GAN = 0.0
        if self.opt.lambda_NCE > 0.0:
            if self.opt.nce_idt:
                self.loss_NCE = self.calculate_NCE_loss(
                    self.feats_l_fakeB, self.feats_l_realA
                )
                self.loss_NCE_Y = self.calculate_NCE_loss(
                    self.feats_l_idtB, self.feats_l_realB
                )
                loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
            else:
                self.loss_NCE = self.calculate_NCE_loss(
                    self.feats_l_fakeB, self.feats_realA
                )
                self.loss_NCE_Y = 0.0
                loss_NCE_both = self.loss_NCE
        else:
            self.loss_NCE, self.loss_NCE_bd, self.loss_NCE_Y, loss_NCE_both = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
        if self.opt.lambda_OA > 0:
            self.loss_OA = (
                self.calculate_OA_NCE_loss(
                    self.feats_m_realA, self.feats_m_fakeB, self.feats_m_realB
                )
                * self.opt.lambda_OA
            )
        else:
            self.loss_OA = 0.0
        if self.opt.lambda_AI > 0:
            self.loss_AI = (
                self.calculate_AI_NCE_loss(
                    [
                        self.feats_i_realA,
                        self.refined_feats_i_realA,
                        self.feats_i_realB,
                    ],
                    [self.feats_m_realA, self.feats_m_realB],
                )
                * self.opt.lambda_AI
            )
        else:
            self.loss_AI = 0.0
        if self.opt.lambda_idt > 0:
            self.loss_idt = (
                self.criterionL1(self.feats_g_fakeB, self.feats_g_realB.detach())
                * self.opt.lambda_idt
            )
        else:
            self.loss_idt = 0.0
        if self.opt.lambda_idt_B > 0:
            self.loss_idt_B = (
                self.criterionL1(self.real_B, self.idt_B) * self.opt.lambda_idt_B
            )
        else:
            self.loss_idt_B = 0.0
        if self.opt.lambda_L1 > 0:
            self.loss_L1 = (
                self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            )
        else:
            self.loss_L1 = 0.0
        self.loss_G = (
            self.loss_G_GAN
            + loss_NCE_both
            + self.loss_AI
            + self.loss_OA
            + self.loss_idt
            + self.loss_L1
        )
        return self.loss_G

    def calculate_NCE_loss(self, feat_q, feat_k):
        n_blocks = len(self.nce_blocks)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_block in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_blocks
        ):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_blocks

    def calculate_OA_NCE_loss(self, feat_src, feat_gen, feat_ref):
        B, _, _ = feat_gen.shape
        feat_B = torch.cat((feat_gen, feat_ref), dim=0)
        feat_src, feat_B = self.netF.forward_oa([feat_src, feat_B])
        feat_gen = feat_B[:B]
        feat_ref = feat_B[B:]
        total_nce_loss = self.criterionOA(feat_src, feat_gen, feat_ref)
        return total_nce_loss

    def calculate_AI_NCE_loss(self, feat_q, feat_k):
        feat_q = self.netF.forward_i(feat_q)
        feat_k = self.netF.forward_m(feat_k)
        total_nce_loss = 0.0
        n = 0
        for f_q in feat_q[:2]:
            for f_k in feat_k:
                loss = self.criterionAI(f_q, f_k)
                total_nce_loss += loss
                n += 1
        total_nce_loss += self.criterionAI(feat_q[2], feat_k[-1])
        n += 1
        return total_nce_loss / n

    def calculate_CC_loss(self, feat_q, feat_k, norm=False):
        feat_k = feat_k.detach()
        if norm:
            feat_q = torch.nn.functional.normalize(feat_q, p=2, dim=1)
            feat_k = torch.nn.functional.normalize(feat_k, p=2, dim=1)
        corr_matrix_q = torch.mm(feat_q, feat_q.t())
        corr_matrix_k = torch.mm(feat_k, feat_k.t())
        cc_loss = torch.nn.functional.l1_loss(corr_matrix_q, corr_matrix_k)
        return cc_loss

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        visual_ret["real_A"] = rearrange(
            self.real_A, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.k, p2=self.k
        )
        visual_ret["real_B"] = rearrange(
            self.real_B, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.k, p2=self.k
        )
        visual_ret["fake_B"] = rearrange(
            self.fake_B, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.k, p2=self.k
        )
        visual_ret["idt_B"] = rearrange(
            self.idt_B, "(b p1 p2) c h w -> b c (p1 h) (p2 w)", p1=self.k, p2=self.k
        )
        return visual_ret

    def test(self):
        with torch.no_grad():
            self.forward()

    def load_networks(
        self,
        epoch,
    ):
        for name in self.model_names:
            if isinstance(name, str):
                if name == "D":
                    continue
                load_filename = "%s_net_%s.pth" % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(
                        self.opt.checkpoints_dir, self.opt.pretrained_name
                    )
                else:
                    load_dir = self.save_dir
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, "net" + name)
                print("loading the model from %s" % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata
                current_model_dict = net.state_dict()
                new_state_dict = {}
                for key, param in state_dict.items():
                    if key in current_model_dict:
                        target_shape = current_model_dict[key].shape
                        if param.shape != target_shape:
                            if param.numel() == current_model_dict[key].numel():
                                print(
                                    f"Adapting weight {key}: {param.shape} -> {target_shape}"
                                )
                                new_state_dict[key] = param.view(target_shape)
                            else:
                                print(
                                    f"Warning: Cannot adapt {key} due to size mismatch."
                                )
                                continue
                        else:
                            new_state_dict[key] = param
                    else:
                        print(f"Key {key} not found in model, skipping.")
                net.load_state_dict(new_state_dict, strict=False)

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
