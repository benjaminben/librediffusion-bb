"""
Model-side concerns of the combined UNet+ControlNet exporter:
  - CombinedUNetControlNet wrapper module
  - HF model loading + optional distillation LoRA fusion
  - Dummy input construction for ONNX trace

Kept separate from the ONNX/TRT graph-build code so changes here
(SD-1.5 vs SD-Turbo, new LoRAs, scheduler tweaks) don't have to scroll
past hundreds of lines of TRT plumbing.
"""

import argparse

import torch
import torch.nn as nn


class CombinedUNetControlNet(nn.Module):
    """Wraps base UNet + ControlNet into a single forward pass.

    Inputs match the standard UNet plus two extras:
      - control_image       [B, 3, H, W]  fp16
      - controlnet_strength [1]           fp16  (scalar multiplier)
    """

    def __init__(self, unet: nn.Module, controlnet: nn.Module) -> None:
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_image: torch.Tensor,
        controlnet_strength: torch.Tensor,
    ) -> torch.Tensor:
        # ControlNet returns a list of down-block residuals + a mid-block residual.
        cn_out = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_image,
            return_dict=False,
        )
        down_residuals, mid_residual = cn_out

        # Apply runtime strength. Cast scale to the residual dtype (fp16) so
        # the Mul stays in fp16 — sample is fp32 to match train-lora.py's
        # plain unet engine, but residuals come out of ControlNet in fp16
        # and we want the skip-connection Add to fuse cleanly.
        scale = controlnet_strength.to(mid_residual.dtype).reshape(())
        down_residuals = [r * scale for r in down_residuals]
        mid_residual = mid_residual * scale

        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]
        return out


def load_models(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    print(f"[load] base UNet:   {args.base_model}")
    print(f"[load] ControlNet:  {args.controlnet}")
    from diffusers import UNet2DConditionModel, ControlNetModel
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=dtype
    ).to(device).eval()
    cn = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype).to(device).eval()
    if args.distillation_lora:
        fuse_distillation_lora(unet, args)
    return unet, cn


def fuse_distillation_lora(unet, args: argparse.Namespace) -> None:
    """Load a distillation LoRA into the UNet and bake it into the base
    weights so the exported ONNX is a single fused graph -- no LoRA adapters
    survive into TRT.

    Pattern: load_lora_adapter -> fuse_lora -> unload_lora. After this
    sequence the UNet looks like a regular UNet whose weights have been
    LoRA-modified; the adapter modules are gone.
    """
    print(f"[lora] loading distillation LoRA: {args.distillation_lora}"
          + (f" :: {args.distillation_lora_weight_name}"
             if args.distillation_lora_weight_name else ""))
    load_kwargs = {}
    if args.distillation_lora_weight_name:
        load_kwargs["weight_name"] = args.distillation_lora_weight_name
    unet.load_lora_adapter(args.distillation_lora, **load_kwargs)
    print("[lora] fusing into UNet weights")
    unet.fuse_lora()
    unet.unload_lora()
    print("[lora] fused; adapter modules unloaded")


def make_dummy_inputs(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    b, h, w = args.batch_size, args.height, args.width
    lh, lw = h // 8, w // 8
    # sample MUST be fp32 to match the plain unet.engine (train-lora.py via
    # streamdiffusion hardcodes torch.float32 here regardless of fp16 mode).
    # The librediffusion C++ wrapper allocates a fp32 sample_buffer_ and
    # binds it to the engine's "sample" input — if this is exported as fp16
    # the wrapper feeds the engine garbage bytes, producing blue-noise
    # output that's invariant to ControlNet inputs.
    sample = torch.randn(b, 4, lh, lw, device=device, dtype=torch.float32)
    timestep = torch.tensor([1.0] * b, device=device, dtype=torch.float32)
    encoder = torch.randn(b, args.text_seq_len, args.text_hidden_dim, device=device, dtype=dtype)
    # IMPORTANT: avoid trivial dummy values. With control_image=zeros + strength=1
    # and torch.onnx do_constant_folding=True, the optimizer can prune the
    # ControlNet branch and fold away the residual*scale Muls -- the engine
    # ends up with declared but disconnected control_image / controlnet_strength
    # inputs, so runtime output is invariant to those bindings.
    control_image = torch.rand(b, 3, h, w, device=device, dtype=dtype)
    strength = torch.tensor([0.5], device=device, dtype=dtype)
    return sample, timestep, encoder, control_image, strength
