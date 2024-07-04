# pylint: disable=E1101,C0415,W0718,R0801
# scripts/train_stage1.py
"""
This is the main training script for stage 1 of the project. 
It imports necessary packages, defines necessary classes and functions, and trains the model using the provided configuration.

The script includes the following classes and functions:

1. Net: A PyTorch model that takes noisy latents, timesteps, reference image latents, face embeddings, 
   and face masks as input and returns the denoised latents.
3. log_validation: A function that logs the validation information using the given VAE, image encoder, 
   network, scheduler, accelerator, width, height, and configuration.
4. train_stage1_process: A function that processes the training stage 1 using the given configuration.

The script also includes the necessary imports and a brief description of the purpose of the file.
"""

import argparse
import copy
import logging
import math
import os
import random
import warnings
from datetime import datetime

import cv2
import diffusers
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from hallo.animate.face_animate_static import StaticPipeline
from hallo.datasets.mask_image import FaceMaskDataset
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.mutual_self_attention import ReferenceAttentionControl
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, move_final_checkpoint,
                              save_checkpoint, seed_everything)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel, 
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        face_mask (torch.Tensor): The face mask tensor.
        uncond_fwd (bool): A flag indicating whether to perform unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """

    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
        imageproj: ImageProjModel,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        face_emb,
        face_mask,
        uncond_fwd: bool = False,
    ):
        """
        Forward pass of the model.
        Args:
            self (Net): The model instance.
            noisy_latents (torch.Tensor): Noisy latents.
            timesteps (torch.Tensor): Timesteps.
            ref_image_latents (torch.Tensor): Reference image latents.
            face_emb (torch.Tensor): Face embedding.
            face_mask (torch.Tensor): Face mask.
            uncond_fwd (bool, optional): Unconditional forward pass. Defaults to False.

        Returns:
            torch.Tensor: Model prediction.
        """

        face_emb = self.imageproj(face_emb)
        face_mask = face_mask.to(device="cuda")
        face_mask_feature = self.face_locator(face_mask)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            mask_cond_fea=face_mask_feature,
            encoder_hidden_states=face_emb,
        ).sample

        return model_pred


def get_noise_scheduler(cfg: argparse.Namespace):
    """
    Create noise scheduler for training

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.

    Returns:
        train noise scheduler and val noise scheduler
    """
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def log_validation(
    vae,
    net,
    scheduler,
    accelerator,
    width,
    height,
    imageproj,
    cfg,
    save_dir,
    global_step,
    face_analysis_model_path,
):
    """
    Log validation generation image.

    Args:
        vae (nn.Module): Variational Autoencoder model.
        net (Net): Main model.
        scheduler (diffusers.SchedulerMixin): Noise scheduler.
        accelerator (accelerate.Accelerator): Accelerator for training.
        width (int): Width of the input images.
        height (int): Height of the input images.
        imageproj (nn.Module): Image projection model.
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
        save_dir (str): directory path to save log result.
        global_step (int): Global step number.

    Returns:
        None
    """
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    ori_net = copy.deepcopy(ori_net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    face_locator = ori_net.face_locator

    generator = torch.manual_seed(42)
    image_enc = FaceAnalysis(
        name="",
        root=face_analysis_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    image_enc.prepare(ctx_id=0, det_size=(640, 640))

    pipe = StaticPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        face_locator=face_locator,
        scheduler=scheduler,
        imageproj=imageproj,
    )

    pil_images = []
    for ref_image_path, mask_image_path in zip(cfg.ref_image_paths, cfg.mask_image_paths):
        # for mask_image_path in mask_image_paths:
        mask_name = os.path.splitext(
            os.path.basename(mask_image_path))[0]
        ref_name = os.path.splitext(
            os.path.basename(ref_image_path))[0]
        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        mask_image_pil = Image.open(mask_image_path).convert("RGB")

        # Prepare face embeds
        face_info = image_enc.get(
            cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (
            x['bbox'][3] - x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = torch.tensor(face_info['embedding'])
        face_emb = face_emb.to(
            imageproj.device, imageproj.dtype)

        image = pipe(
            ref_image_pil,
            mask_image_pil,
            width,
            height,
            20,
            3.5,
            face_emb,
            generator=generator,
        ).images
        image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        # Save ref_image, src_image and the generated_image
        w, h = res_image_pil.size
        canvas = Image.new("RGB", (w * 3, h), "white")
        ref_image_pil = ref_image_pil.resize((w, h))
        mask_image_pil = mask_image_pil.resize((w, h))
        canvas.paste(ref_image_pil, (0, 0))
        canvas.paste(mask_image_pil, (w, 0))
        canvas.paste(res_image_pil, (w * 2, 0))

        out_file = os.path.join(
            save_dir, f"{global_step:06d}-{ref_name}_{mask_name}.jpg"
        )
        canvas.save(out_file)

    del pipe
    del ori_net
    torch.cuda.empty_cache()

    return pil_images


def train_stage1_process(cfg: argparse.Namespace) -> None:
    """
    Trains the model using the given configuration (cfg).

    Args:
        cfg (dict): The configuration dictionary containing the parameters for training.

    Notes:
        - This function trains the model using the given configuration.
        - It initializes the necessary components for training, such as the pipeline, optimizer, and scheduler.
        - The training progress is logged and tracked using the accelerator.
        - The trained model is saved after the training is completed.
    """
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # create output dir for training
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    module_dir = os.path.join(save_dir, "modules")
    validation_dir = os.path.join(save_dir, "validation")

    if accelerator.is_main_process:
        init_output_dir([save_dir, checkpoint_dir, module_dir, validation_dir])

    accelerator.wait_for_everyone()

    # create model
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    # create model
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
        use_landmark=False
    ).to(device="cuda", dtype=weight_dtype)
    imageproj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ).to(device="cuda", dtype=weight_dtype)

    if cfg.face_locator_pretrained:
        face_locator = FaceLocator(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda", dtype=weight_dtype)
        miss, _ = face_locator.load_state_dict(
            cfg.face_state_dict_path, strict=False)
        logger.info(f"Missing key for face locator: {len(miss)}")
    else:
        face_locator = FaceLocator(
            conditioning_embedding_channels=320,
        ).to(device="cuda", dtype=weight_dtype)
    # Freeze
    vae.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    reference_unet.requires_grad_(True)
    imageproj.requires_grad_(True)
    face_locator.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
    ).to(dtype=weight_dtype)

    # get noise scheduler
    train_noise_scheduler, val_noise_scheduler = get_noise_scheduler(cfg)

    # init optimizer
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(
        filter(lambda p: p.requires_grad, net.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # init scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # get data loader
    train_dataset = FaceMaskDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        data_meta_paths=cfg.data.meta_paths,
        sample_margin=cfg.data.sample_margin,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=4
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        logger.info(f"save config to {save_dir}")
        OmegaConf.save(
            cfg, os.path.join(save_dir, "config.yaml")
        )
    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # load checkpoint
    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        global_step = load_checkpoint(cfg, checkpoint_dir, accelerator)
        first_epoch = global_step // num_update_steps_per_epoch

       # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")
    net.train()
    for _ in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["img"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215

                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                face_mask_img = batch["tgt_mask"]
                face_mask_img = face_mask_img.unsqueeze(
                    2)
                face_mask_img = face_mask_img.to(weight_dtype)

                uncond_fwd = random.random() < cfg.uncond_ratio
                face_emb_list = []
                ref_image_list = []
                for _, (ref_img, face_emb) in enumerate(
                    zip(batch["ref_img"], batch["face_emb"])
                ):
                    if uncond_fwd:
                        face_emb_list.append(torch.zeros_like(face_emb))
                    else:
                        face_emb_list.append(face_emb)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()
                    ref_image_latents = ref_image_latents * 0.18215

                    face_emb = torch.stack(face_emb_list, dim=0).to(
                        dtype=imageproj.dtype, device=imageproj.device
                    )

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    face_emb,
                    face_mask_img,
                    uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0 or global_step == cfg.solver.max_train_steps:
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(
                        checkpoint_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        delete_additional_ckpt(checkpoint_dir, 3)
                    accelerator.save_state(save_path)
                    accelerator.wait_for_everyone()
                    unwrap_net = accelerator.unwrap_model(net)
                    if accelerator.is_main_process:
                        save_checkpoint(
                            unwrap_net.reference_unet,
                            module_dir,
                            "reference_unet",
                            global_step,
                            total_limit=3,
                        )
                        save_checkpoint(
                            unwrap_net.imageproj,
                            module_dir,
                            "imageproj",
                            global_step,
                            total_limit=3,
                        )
                        save_checkpoint(
                            unwrap_net.denoising_unet,
                            module_dir,
                            "denoising_unet",
                            global_step,
                            total_limit=3,
                        )
                        save_checkpoint(
                            unwrap_net.face_locator,
                            module_dir,
                            "face_locator",
                            global_step,
                            total_limit=3,
                        )

                if global_step % cfg.val.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)
                        log_validation(
                            vae=vae,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            imageproj=imageproj,
                            cfg=cfg,
                            save_dir=validation_dir,
                            global_step=global_step,
                            face_analysis_model_path=cfg.face_analysis_model_path
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                # process final module weight for stage2
                if accelerator.is_main_process:
                    move_final_checkpoint(save_dir, module_dir, "reference_unet")
                    move_final_checkpoint(save_dir, module_dir, "imageproj")
                    move_final_checkpoint(save_dir, module_dir, "denoising_unet")
                    move_final_checkpoint(save_dir, module_dir, "face_locator")
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        train_stage1_process(config)
    except Exception as e:
        logging.error("Failed to execute the training process: %s", e)
