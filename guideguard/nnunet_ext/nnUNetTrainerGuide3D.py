import os
from os.path import join

import torch
from torch import autocast
from torch._dynamo import OptimizedModule

import nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae as _mae
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

_mae.dummy_context = dummy_context

N_OUT = 4
N_IMAGE_CHANNELS = 3


def decode_code(code):
    c = torch.round(code).to(torch.int32)
    brain = (c & 1).to(torch.float32)
    regions = torch.cat([(c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1], dim=1).to(torch.float32)
    return brain, regions


def masked_l1_mse(pred, target, mask):
    n = mask.sum().clamp_min(1.0)
    d = pred - target
    return (d.abs() * mask).sum() / n, (d * d * mask).sum() / n


def expand_state_dict(dst_sd, src_sd, seg_init_scale=0.1):
    out = {}
    for k, v in dst_sd.items():
        s = src_sd[k].to(device=v.device, dtype=v.dtype)
        if s.shape == v.shape:
            out[k] = s.clone()
        elif v.ndim == 5 and s.ndim == 5 and v.shape[0] == s.shape[0] and v.shape[2:] == s.shape[2:] \
                and v.shape[1] > s.shape[1]:
            t = torch.zeros_like(v)
            t[:, :s.shape[1]] = s
            out[k] = t
        elif ".seg_layers." in k and v.shape[0] > s.shape[0] and v.shape[1:] == s.shape[1:]:
            t = v.clone() * (seg_init_scale if v.ndim > 1 else 0.0)
            t[:s.shape[0]] = s
            out[k] = t
        else:
            raise ValueError(f"cannot map {k}: {tuple(s.shape)} -> {tuple(v.shape)}")
    extra = sorted(set(src_sd) - set(dst_sd))
    if extra:
        raise KeyError(f"pretrained keys not in the new network: {extra[:5]}")
    return out


class nnUNetTrainerGuide3D(_mae.nnUNetTrainerMRCT_mae):
    EPOCHS = 400
    SNAPSHOT_EVERY = 50
    INITIAL_LR = 1e-4
    MSE_WEIGHT = 5.0
    SEG_WEIGHT = 0.05
    FLIP_P = 0.5
    FLIP_DIM = -1
    PRIOR_DROPOUT = 0.15
    SEG_INIT_SCALE = 0.1

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = self.INITIAL_LR
        self.num_epochs = self.EPOCHS

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return get_network_from_plans(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                      num_input_channels, N_OUT, allow_init=True,
                                      deep_supervision=enable_deep_supervision)

    def _module(self):
        mod = self.network.module if self.is_ddp else self.network
        return mod._orig_mod if isinstance(mod, OptimizedModule) else mod

    def initialize(self):
        super().initialize()
        path = os.environ.get("GUIDE3D_INIT")
        if not path:
            raise RuntimeError("set GUIDE3D_INIT to the nnU-Net-L1 checkpoint to fine-tune from")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mod = self._module()
        mod.load_state_dict(expand_state_dict(mod.state_dict(), ck["network_weights"], self.SEG_INIT_SCALE),
                            strict=True)
        self.print_to_log_file(f"initialised from {path}")

    def _build_loss(self):
        return DC_and_BCE_loss({}, {"batch_dice": self.configuration_manager.batch_dice, "do_bg": True,
                                    "smooth": 1e-5, "ddp": self.is_ddp}, use_ignore_label=False,
                               dice_class=MemoryEfficientSoftDiceLoss)

    def _losses(self, output, target):
        o = output.float()
        t1c = target[:, 0:1].float()
        brain, regions = decode_code(target[:, 1:2])
        l1, mse = masked_l1_mse(o[:, 0:1], t1c, brain)
        seg = self.loss(o[:, 1:], regions)
        return l1 + self.MSE_WEIGHT * mse + self.SEG_WEIGHT * seg, {"l1": l1.detach(), "mse": mse.detach(),
                                                                    "seg": seg.detach()}

    def _augment(self, data, target):
        b = data.shape[0]
        flip = torch.rand(b) < self.FLIP_P
        if flip.any():
            idx = flip.nonzero().flatten().to(data.device)
            data[idx] = torch.flip(data[idx], dims=(self.FLIP_DIM,))
            target[idx] = torch.flip(target[idx], dims=(self.FLIP_DIM,))
        if data.shape[1] > N_IMAGE_CHANNELS and self.PRIOR_DROPOUT > 0:
            drop = torch.rand(b) < self.PRIOR_DROPOUT
            if drop.any():
                data[drop.nonzero().flatten().to(data.device), N_IMAGE_CHANNELS:] = 0
        return data, target

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        data, target = self._augment(data, target)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
        total, parts = self._losses(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": total.detach().cpu().numpy(), **{k: v.cpu().numpy() for k, v in parts.items()}}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
        _, parts = self._losses(output, target)
        return {"loss": (parts["l1"] + self.MSE_WEIGHT * parts["mse"]).cpu().numpy(),
                **{k: v.cpu().numpy() for k, v in parts.items()}}

    def on_epoch_end(self):
        ep = self.current_epoch
        super().on_epoch_end()
        if self.SNAPSHOT_EVERY > 0 and (ep + 1) % self.SNAPSHOT_EVERY == 0 and ep != self.num_epochs - 1:
            cur = self.current_epoch
            self.current_epoch = ep
            try:
                self.save_checkpoint(join(self.output_folder, f"checkpoint_ep{ep + 1:04d}.pth"))
            finally:
                self.current_epoch = cur

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("regression task: select checkpoints with scripts/select_guide3d.py")
