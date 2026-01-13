"""
Test script for ProstNFound-RL models

This is adapted from test.py to support RL models with attention mechanisms.
"""

import argparse
from collections import defaultdict
import json
import os
from argparse import ArgumentParser, Namespace
import time

import PIL
import hydra
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
import rich_argparse
import torch
from PIL import Image
import medAI
from medAI.layers.masked_prediction_module import get_bags_of_predictions
from medAI.modeling.prostnfound_rl import ProstNFoundRL
try:
    from medAI.modeling.prostnfound_rl_v2 import ProstNFoundRLV2
except ImportError:
    ProstNFoundRLV2 = None
from medAI.utils.accumulators import DataFrameCollector
from medAI.utils.argparse import UpdateDictAction

import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm

from medAI.datasets.nct2013 import data_accessor
from medAI.modeling import list_models, create_model
from src.loss import MaskedPredictionModule
from src.loaders import get_dataloaders
from src.evaluator import show_heatmap_prediction
from src.utils import render_heatmap
import skimage
from train_rl import ProstNFoundMeta
from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator


@hydra.main(config_path="cfg", config_name="test_rl", version_base="1.3")
def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    else: 
        state = None

    train_args = Namespace(**state["args"])

    # IMPORTANT: Always use the model config from the checkpoint
    # This ensures we load the correct RL model architecture
    print(f"Checkpoint model: {train_args.model}")
    print(f"Checkpoint model_kw: {train_args.model_kw}")
    
    # Override test config with checkpoint's model config
    args.model = train_args.model
    # Convert to plain dict to allow modification (OmegaConf struct mode blocks new keys)
    model_kw = dict(train_args.model_kw)
    
    # Detect legacy checkpoints that don't have policy_arch_version
    # Legacy checkpoints used a simpler attention_map_head architecture
    # NOTE: policy_arch_version is only for V1 RL models (prostnfound_rl), NOT V2 models (prostnfound_rl_v2)
    is_v1_model = 'rl_v2' not in args.model  # V1 models don't have 'rl_v2' in their name
    
    if is_v1_model and 'policy_arch_version' not in model_kw:
        # Check if this is a legacy checkpoint by inspecting the state dict
        has_legacy_arch = False
        for key in state.get("model", {}).keys():
            # Legacy architecture has attention_map_head.3.weight with shape [1, 256, 1, 1]
            # New architecture has attention_map_head.3.weight with shape [128, 256, 3, 3]
            if "attention_map_head.3.weight" in key:
                weight_shape = state["model"][key].shape
                if weight_shape == torch.Size([1, 256, 1, 1]):
                    has_legacy_arch = True
                    break
        
        if has_legacy_arch:
            print("Detected legacy V1 architecture checkpoint. Setting policy_arch_version='v1'")
            model_kw['policy_arch_version'] = 'v1'
        else:
            print("V1 architecture version not in checkpoint. Defaulting to 'v2'")
            model_kw['policy_arch_version'] = 'v2'
    elif not is_v1_model:
        # V2 models don't use policy_arch_version - remove it if present
        if 'policy_arch_version' in model_kw:
            print("Removing policy_arch_version from V2 model kwargs (not applicable)")
            del model_kw['policy_arch_version']
    
    # Update args with the modified model_kw (use OmegaConf.update to handle struct mode)
    OmegaConf.set_struct(args, False)  # Disable struct mode temporarily
    args.model_kw = model_kw
    OmegaConf.set_struct(args, True)  # Re-enable struct mode

    if args.save_checkpoint:
        torch.save(state, os.path.join(args.output_dir, "checkpoint.pth"))

    # Saving test-time config is always safe
    OmegaConf.save(args, os.path.join(args.output_dir, "test_args.yaml"))

    # Older training configs saved as plain dicts/Namespaces can contain
    # objects OmegaConf cannot serialize (e.g. types, unions). Since this
    # is only for bookkeeping, fail softly if saving them doesn't work.
    try:
        OmegaConf.save(
            state["args"],
            os.path.join(args.output_dir, "train_args.yaml"),
        )
    except Exception as e:
        print(f"Warning: could not save train_args.yaml: {e}")

    # Create model and detect if it's RL
    base_model = create_model(args.model, **args.model_kw)
    # Check for both V1 (ProstNFoundRL) and V2 (ProstNFoundRLV2) models
    is_rl_model = isinstance(base_model, ProstNFoundRL)
    if ProstNFoundRLV2 is not None:
        is_rl_model = is_rl_model or isinstance(base_model, ProstNFoundRLV2)
    print(f"Model type: {type(base_model)}")
    print(f"Is RL model: {is_rl_model}")
    
    model = ProstNFoundMeta(base_model, is_rl=is_rl_model)
    print(model.load_state_dict(state["model"], strict=False))
    model.to(device)
    model.eval()
    if args.torch_compile:
        model = torch.compile(model)

    if args.get("data"):
        loaders = get_dataloaders(args.data, mode="test")
    elif "data" in vars(train_args):
        loaders = get_dataloaders(train_args.data, mode="test")
    else:
        loaders = get_dataloaders(train_args, mode="test")

    # maybe calibrate the temperature and bias of the model
    if args.calibration_mode == "pixel":
        do_calibration_pixel_wise_balanced_bce(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )
    elif args.calibration_mode == "bag":
        do_calibration_bag_wise(
            model, loaders, args.calibrate_bias, args.calibrate_temperature
        )

    evaluator = Evaluator(
        log_images=False, include_patient_metrics=args.get('include_patient_metrics', False)
    )
    accumulator = defaultdict(list)
    
    # For RL models, also accumulate attention point statistics
    if is_rl_model:
        rl_accumulator = defaultdict(list)
        # For analysis: store per-core RL attention probabilities and coordinates
        rl_attention_point_records = []
        # Track high prediction cases (for verification that model predicts cancer)
        high_prediction_cases = []

    loader = loaders[args.split]

    # warmup
    for _ in range(10):
        batch = next(iter(loader))
        if is_rl_model:
            model(batch, deterministic=True)
        else:
            model(batch)

    for i, data in enumerate(tqdm(loader)):

        # measure inference
        t0 = time.perf_counter()

        with torch.amp.autocast_mode.autocast(
            device_type=device.type, enabled=args.use_amp
        ):
            with torch.inference_mode():
                # Use deterministic policy for RL models
                if is_rl_model:
                    data = model(data, deterministic=True)
                else:
                    data = model(data)

        if args.postprocess:
            cancer_logits = data.pop("cancer_logits")
            heatmap = cancer_logits[0, 0].sigmoid().cpu().numpy()
            # blur and upsample
            
            import skimage

            # Apply Gaussian blur (operates on float values in [0, 1])
            blurred = skimage.filters.gaussian(heatmap, sigma=1.5)
            # Resize (returns values in [0, 1] range)
            upsampled = skimage.transform.resize(blurred, (256, 256), order=1, anti_aliasing=True)
            # Keep as float in [0, 1] range
            heatmap = upsampled
            data["cancer_probs"] = torch.tensor(heatmap)[None, None, ...]
        else:
            # get raw heatmap and also save as png
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = Image.fromarray(heatmap)

        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator["infer_time"].append(infer_time)
        
        # Store RL-specific information
        if is_rl_model:
            # For discrete mode: store attention coordinates
            if 'rl_attention_coords' in data and data['rl_attention_coords'] is not None:
                rl_accumulator['attention_coords'].append(data['rl_attention_coords'].cpu())
            # For both discrete and continuous: store attention maps
            if 'rl_attention_map' in data and data['rl_attention_map'] is not None:
                rl_accumulator['attention_maps'].append(data['rl_attention_map'].cpu())
            # Store masks for Needle Focus Ratio metric
            if 'needle_mask' in data:
                rl_accumulator['needle_mask'].append(data['needle_mask'].cpu())
            if 'prostate_mask' in data:
                rl_accumulator['prostate_mask'].append(data['prostate_mask'].cpu())

        if args.save_raw_heatmaps:
            # get raw heatmap and also save as png
            heatmap = Image.fromarray(heatmap)
            os.makedirs(os.path.join(args.output_dir, "raw_heatmaps"), exist_ok=True)
            heatmap.save(
                os.path.join(
                    args.output_dir, "raw_heatmaps", data["core_id"][0] + ".png"
                )
            )

        patient_id = data['patient_id'][0]
        core_id = data['core_id'][0]
    
        output_file = os.path.join(
            args.output_dir, 
            "heatmaps", 
            patient_id,
            f"{core_id}.{args.save_format}"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Get the image (after preprocessing, it's 256x256)
        bmode = data["bmode"][0].cpu()  # (C, H, W)
        if bmode.shape[0] == 3:  # RGB, convert to grayscale
            bmode = bmode.mean(dim=0)
        bmode_np = bmode.numpy()
        img_h, img_w = bmode_np.shape  # These will be 256x256 (square after preprocessing)
        
        # Get ORIGINAL aspect ratio from metadata if available
        # Original images are not square - they have different aspect ratios
        original_aspect_ratio = None
        if "image_height_mm" in data and "image_width_mm" in data:
            height_mm = float(data["image_height_mm"][0].item())
            width_mm = float(data["image_width_mm"][0].item())
            if height_mm > 0 and width_mm > 0:
                original_aspect_ratio = width_mm / height_mm  # width/height
        elif "info" in data and isinstance(data["info"], dict):
            # Try to get from info dict if available
            info = data["info"][0] if isinstance(data["info"], list) else data["info"]
            if isinstance(info, dict):
                height_mm = info.get("heightMm")
                width_mm = info.get("widthMm")
                if height_mm is not None and width_mm is not None:
                    height_mm = float(height_mm)
                    width_mm = float(width_mm)
                    if height_mm > 0 and width_mm > 0:
                        original_aspect_ratio = width_mm / height_mm
        
        # Use original aspect ratio if available, otherwise use processed image ratio (1.0 for square)
        if original_aspect_ratio is not None:
            display_aspect_ratio = original_aspect_ratio
        else:
            display_aspect_ratio = img_w / img_h  # Will be 1.0 for 256x256
        
        # Detect if we need 3-panel layout (for continuous V2 models)
        is_v2_model = False
        is_continuous_v2 = False
        try:
            if hasattr(base_model, 'discrete_attention'):
                is_v2_model = True
                is_continuous_v2 = not base_model.discrete_attention
        except:
            pass
        
        # Calculate figure size preserving original aspect ratio
        fig_height = 5
        # Width for panels side by side, preserving aspect ratio
        num_panels = 3 if (is_rl_model and is_continuous_v2) else 2
        fig_width = fig_height * display_aspect_ratio * num_panels + 1
        fig, ax = plt.subplots(1, num_panels, figsize=(fig_width, fig_height))
        
        # Get masks
        prostate_mask = data["prostate_mask"][0, 0].cpu().numpy() if "prostate_mask" in data else None
        needle_mask = data["needle_mask"][0, 0].cpu().numpy() if "needle_mask" in data else None
        
        # Get heatmap
        if "cancer_probs" in data:
            heatmap = data["cancer_probs"][0, 0].cpu().numpy()
        elif "cancer_logits" in data:
            heatmap = data["cancer_logits"][0, 0].sigmoid().cpu().numpy()
        else:
            heatmap = None
        
        # Panel 0: B-mode image with contours
        # Use original aspect ratio for display (not the square 256x256)
        if original_aspect_ratio is not None:
            ax[0].imshow(bmode_np, cmap='gray', aspect=1.0/display_aspect_ratio)
        else:
            ax[0].imshow(bmode_np, cmap='gray', aspect='auto')
        if prostate_mask is not None:
            if prostate_mask.shape != bmode_np.shape:
                prostate_mask = skimage.transform.resize(
                    prostate_mask, bmode_np.shape, preserve_range=True, order=0
                ).astype(bool)
            ax[0].contour(prostate_mask, colors='white', alpha=0.7, linewidths=1.5, levels=[0.5])
        if needle_mask is not None:
            if needle_mask.shape != bmode_np.shape:
                needle_mask = skimage.transform.resize(
                    needle_mask, bmode_np.shape, preserve_range=True, order=0
                ).astype(bool)
            ax[0].contour(needle_mask, colors='yellow', alpha=0.7, linewidths=1.5, levels=[0.5])
        ax[0].set_title('B-mode Image', fontsize=11, pad=5)
        ax[0].axis('off')
        
        # Panel 1 (middle): RL Attention Map OR Decoder Heatmap (depending on model type)
        # For continuous V2: Show RL attention, For others: Show decoder heatmap
        if is_rl_model and is_continuous_v2:
            # Middle panel: RL Attention Map
            if 'rl_attention_map' in data and data['rl_attention_map'] is not None:
                attn_map = data['rl_attention_map'][0].detach().cpu().numpy()
                attn_map = np.squeeze(attn_map)
                if attn_map.ndim == 2:
                    # Resize to image size
                    attn_map_resized = skimage.transform.resize(
                        attn_map, (img_h, img_w), order=1, preserve_range=True
                    )
                    # Normalize to [0, 1]
                    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
                    
                    if original_aspect_ratio is not None:
                        im = ax[1].imshow(attn_map_resized, cmap='Reds', vmin=0, vmax=1, aspect=1.0/display_aspect_ratio)
                    else:
                        im = ax[1].imshow(attn_map_resized, cmap='Reds', vmin=0, vmax=1, aspect='auto')
                    
                    # Overlay contours
                    if needle_mask is not None:
                        if needle_mask.shape != bmode_np.shape:
                            needle_mask_display = skimage.transform.resize(
                                needle_mask, bmode_np.shape, preserve_range=True, order=0
                            ).astype(bool)
                        else:
                            needle_mask_display = needle_mask
                        ax[1].contour(needle_mask_display, colors='white', alpha=0.7, linewidths=1.5, levels=[0.5])
                    if prostate_mask is not None:
                        if prostate_mask.shape != bmode_np.shape:
                            prostate_mask_display = skimage.transform.resize(
                                prostate_mask, bmode_np.shape, preserve_range=True, order=0
                            ).astype(bool)
                        else:
                            prostate_mask_display = prostate_mask
                        ax[1].contour(prostate_mask_display, colors='cyan', alpha=0.5, linewidths=1.5, levels=[0.5])
                    
                    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
                    ax[1].set_title('RL Attention Map', fontsize=11, pad=5)
                else:
                    # Fallback
                    if original_aspect_ratio is not None:
                        ax[1].imshow(bmode_np, cmap='gray', aspect=1.0/display_aspect_ratio)
                    else:
                        ax[1].imshow(bmode_np, cmap='gray', aspect='auto')
                    ax[1].set_title('RL Attention (N/A)', fontsize=11, pad=5)
            else:
                # Fallback
                if original_aspect_ratio is not None:
                    ax[1].imshow(bmode_np, cmap='gray', aspect=1.0/display_aspect_ratio)
                else:
                    ax[1].imshow(bmode_np, cmap='gray', aspect='auto')
                ax[1].set_title('RL Attention (N/A)', fontsize=11, pad=5)
            ax[1].axis('off')
        else:
            # For non-continuous or non-RL: Show decoder heatmap in panel 1
            if heatmap is not None:
                # Resize heatmap to match display
                if heatmap.shape != bmode_np.shape:
                    heatmap_resized = skimage.transform.resize(
                        heatmap, bmode_np.shape, preserve_range=True, order=1, anti_aliasing=True
                    )
                else:
                    heatmap_resized = heatmap
                
                # Show heatmap with colormap
                if original_aspect_ratio is not None:
                    im = ax[1].imshow(heatmap_resized, cmap='viridis', vmin=0, vmax=1, aspect=1.0/display_aspect_ratio)
                else:
                    im = ax[1].imshow(heatmap_resized, cmap='viridis', vmin=0, vmax=1, aspect='auto')
                
                # Overlay contours on heatmap
                if needle_mask is not None:
                    if needle_mask.shape != bmode_np.shape:
                        needle_mask_display = skimage.transform.resize(
                            needle_mask, bmode_np.shape, preserve_range=True, order=0
                        ).astype(bool)
                    else:
                        needle_mask_display = needle_mask
                    ax[1].contour(needle_mask_display, colors='white', alpha=0.7, linewidths=1.5, levels=[0.5])
                if prostate_mask is not None:
                    if prostate_mask.shape != bmode_np.shape:
                        prostate_mask_display = skimage.transform.resize(
                            prostate_mask, bmode_np.shape, preserve_range=True, order=0
                        ).astype(bool)
                    else:
                        prostate_mask_display = prostate_mask
                    ax[1].contour(prostate_mask_display, colors='cyan', alpha=0.5, linewidths=1.5, levels=[0.5])
            else:
                # Fallback if no heatmap
                if original_aspect_ratio is not None:
                    ax[1].imshow(bmode_np, cmap='gray', aspect=1.0/display_aspect_ratio)
                else:
                    ax[1].imshow(bmode_np, cmap='gray', aspect='auto')
            ax[1].set_title('Decoder Heatmap', fontsize=11, pad=5)
            ax[1].axis('off')
        
        # Extract key information for clear display
        gt_label = data["label"][0].item() == 1
        gt_involvement = data["involvement"][0].item()
        
        # Get grade group if available
        grade_group = None
        if "grade_group" in data:
            grade_group = int(data["grade_group"][0].item())
        
        # Get classification score (from classification head)
        cls_score = None
        if "image_level_classification_outputs" in data and data["image_level_classification_outputs"] is not None:
            import torch.nn.functional as F
            cls_logits = data["image_level_classification_outputs"][0]  # (B, num_classes)
            cls_probs = F.softmax(cls_logits, dim=1)
            # Get probability of positive class (csPCa or cancer depending on task)
            if cls_probs.shape[1] > 1:
                cls_score = float(cls_probs[0, 1].item())  # Positive class probability
            else:
                cls_score = float(cls_probs[0, 0].item())
        
        # Get ROI average (heatmap view - average in needle region)
        roi_avg = None
        if "average_needle_heatmap_value" in data:
            roi_avg = float(data["average_needle_heatmap_value"][0].item())
        elif "cancer_logits" in data and "needle_mask" in data:
            # Fallback: compute from logits if average not available
            import torch.nn.functional as F
            logits = data["cancer_logits"]
            needle_mask = data["needle_mask"] > 0.5
            if needle_mask.any():
                roi_avg = float(logits[needle_mask].sigmoid().mean().item())
        
        # Build title with all requested information
        title_parts = []
        title_parts.append(f"GT: {'Cancer' if gt_label else 'Benign'} (Inv: {gt_involvement:.1%})")
        
        if grade_group is not None:
            title_parts.append(f"GG: {grade_group}")
        
        if cls_score is not None:
            title_parts.append(f"Cls: {cls_score:.1%}")
        
        if roi_avg is not None:
            title_parts.append(f"ROI: {roi_avg:.1%}")
        
        title_parts.append(f"Core: {core_id}")
        
        title_text = " | ".join(title_parts)
        fig.suptitle(title_text, fontsize=10, y=0.98)
        
        # Panel 2 (right): Decoder Output Heatmap (for continuous V2)
        # For continuous V2, this is the rightmost panel showing final decoder predictions
        # For other models, overlay attention points on the heatmap panel
        if is_rl_model and is_continuous_v2:
            # Right panel: Decoder output heatmap
            if heatmap is not None:
                # Resize heatmap to match display
                if heatmap.shape != bmode_np.shape:
                    heatmap_resized = skimage.transform.resize(
                        heatmap, bmode_np.shape, preserve_range=True, order=1, anti_aliasing=True
                    )
                else:
                    heatmap_resized = heatmap
                
                # Show heatmap with colormap
                if original_aspect_ratio is not None:
                    im = ax[2].imshow(heatmap_resized, cmap='viridis', vmin=0, vmax=1, aspect=1.0/display_aspect_ratio)
                else:
                    im = ax[2].imshow(heatmap_resized, cmap='viridis', vmin=0, vmax=1, aspect='auto')
                
                # Overlay contours
                if needle_mask is not None:
                    if needle_mask.shape != bmode_np.shape:
                        needle_mask_display = skimage.transform.resize(
                            needle_mask, bmode_np.shape, preserve_range=True, order=0
                        ).astype(bool)
                    else:
                        needle_mask_display = needle_mask
                    ax[2].contour(needle_mask_display, colors='white', alpha=0.7, linewidths=1.5, levels=[0.5])
                if prostate_mask is not None:
                    if prostate_mask.shape != bmode_np.shape:
                        prostate_mask_display = skimage.transform.resize(
                            prostate_mask, bmode_np.shape, preserve_range=True, order=0
                        ).astype(bool)
                    else:
                        prostate_mask_display = prostate_mask
                    ax[2].contour(prostate_mask_display, colors='cyan', alpha=0.5, linewidths=1.5, levels=[0.5])
                
                plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            else:
                # Fallback
                if original_aspect_ratio is not None:
                    ax[2].imshow(bmode_np, cmap='gray', aspect=1.0/display_aspect_ratio)
                else:
                    ax[2].imshow(bmode_np, cmap='gray', aspect='auto')
            ax[2].set_title('Decoder Output', fontsize=11, pad=5)
            ax[2].axis('off')
        
        # Overlay attention points/patches if RL model (for discrete modes)
        if is_rl_model:
            # Detect if V2 model (patch-based)
            is_discrete_v2 = False
            try:
                if hasattr(base_model, 'discrete_attention'):
                    is_discrete_v2 = base_model.discrete_attention
                elif 'rl_action_indices' in data:
                    is_discrete_v2 = True
            except:
                pass
            
            # For discrete modes, overlay attention points/patches on the heatmap panel
            if 'rl_attention_coords' in data and data['rl_attention_coords'] is not None:
                coords = data['rl_attention_coords'][0].cpu().numpy()  # (k, 2) in [x, y]
                
                # Scale coordinates from model space to display pixel space
                try:
                    if hasattr(base_model, 'policy') and hasattr(base_model.policy, 'image_size'):
                        model_image_size = base_model.policy.image_size
                    else:
                        model_image_size = 256
                except:
                    model_image_size = 256
                
                # Scale to pixel coordinates
                scale_x = img_w / model_image_size
                scale_y = img_h / model_image_size
                
                xs = coords[:, 0] * scale_x
                ys = coords[:, 1] * scale_y
                
                # Determine which panel to annotate (panel 1 for 2-panel, panel 2 for 3-panel)
                annot_panel_idx = 2 if (is_continuous_v2 and num_panels == 3) else 1
                
                if is_discrete_v2:
                    # V2 Discrete: Draw patch rectangles
                    # Get patch size from attention map if available
                    patch_size = 16  # Default patch size in pixels
                    if 'rl_attention_map' in data and data['rl_attention_map'] is not None:
                        attn_map = data['rl_attention_map'][0].detach().cpu().numpy()
                        attn_map = np.squeeze(attn_map)
                        if attn_map.ndim == 2:
                            H_patches, W_patches = attn_map.shape
                            patch_size_y = img_h / H_patches
                            patch_size_x = img_w / W_patches
                            patch_size = (patch_size_x + patch_size_y) / 2
                    
                    # Draw rectangles for selected patches
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        # Center the rectangle on the point
                        rect_x = x - patch_size / 2
                        rect_y = y - patch_size / 2
                        rect = Rectangle(
                            (rect_x, rect_y), patch_size, patch_size,
                            linewidth=2.5, edgecolor='red', facecolor='none',
                            linestyle='-', zorder=10
                        )
                        ax[annot_panel_idx].add_patch(rect)
                        # Add patch number
                        ax[annot_panel_idx].text(x, y, f'{i+1}', color='yellow', fontsize=9,
                                 ha='center', va='center', fontweight='bold',
                                 bbox=dict(boxstyle='circle', facecolor='red', alpha=0.7),
                                 zorder=11)
                    
                    ax[annot_panel_idx].legend([f'RL Patches ({len(xs)})'], loc='upper right', fontsize=8, framealpha=0.8)
                else:
                    # V1: Plot attention points
                    ax[annot_panel_idx].scatter(xs, ys, c='red', marker='x',
                                s=150, linewidths=2.5, 
                                label=f'RL Attention ({len(xs)})',
                                zorder=10)
                    
                    # Add legend to heatmap panel
                    ax[annot_panel_idx].legend(loc='upper right', fontsize=8, framealpha=0.8)
                
                point_probs = None

                # If attention map is available, annotate probabilities from it (only for discrete modes with coords)
                if 'rl_attention_map' in data and data['rl_attention_map'] is not None:
                    # rl_attention_map: typically (B, 1, H, W) or (B, H, W)
                    attn_map = data['rl_attention_map'][0].detach().cpu().numpy()
                    # Squeeze any singleton channel dimension
                    attn_map = np.squeeze(attn_map)
                    if attn_map.ndim != 2:
                        raise ValueError(
                            f"Expected attention map to be 2D after squeeze, got shape {attn_map.shape}"
                        )
                    H, W = attn_map.shape

                    # Map coords to attention map space
                    feature_scale_y = H / model_image_size
                    feature_scale_x = W / model_image_size

                    point_probs = []
                    for i, (x, y) in enumerate(coords):  # Use original coords
                        ix = int(np.clip(x * feature_scale_x, 0, W - 1))
                        iy = int(np.clip(y * feature_scale_y, 0, H - 1))
                        p = float(attn_map[iy, ix])
                        point_probs.append(p)
                        
                        # Annotate with scaled pixel coordinates on annotation panel
                        ax[annot_panel_idx].text(
                            xs[i] + img_w*0.02, ys[i] - img_h*0.02,  # Offset proportional to image size
                            f"{p:.2f}",
                            color="yellow",
                            fontsize=7,
                            ha="left",
                            va="top",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                            zorder=11,
                        )

                # Save probabilities for this core into a summary table (if we computed them)
                if point_probs is not None:
                    for j, (x, y, p) in enumerate(zip(coords[:, 0], coords[:, 1], point_probs)):
                        rl_attention_point_records.append(
                            {
                                "patient_id": patient_id,
                                "core_id": core_id,
                                "point_idx": j,
                                "x": float(x),  # Save in model space [0, 256]
                                "y": float(y),  # Save in model space [0, 256]
                                "prob": float(p),
                            }
                        )
        
        plt.tight_layout()
        plt.savefig(
            output_file,
            format=args.save_format,
            bbox_inches='tight',
            dpi=150
        )
        plt.close()
        
        # Track high prediction cases for logging
        if cls_score is not None and cls_score > 0.4:
            high_prediction_cases.append({
                'core_id': core_id,
                'patient_id': patient_id,
                'cls_score': cls_score,
                'roi_avg': roi_avg if roi_avg is not None else 'N/A',
                'gt_label': 'Cancer' if gt_label else 'Benign',
                'gt_involvement': gt_involvement,
                'type': 'CLS'
            })
        elif roi_avg is not None and roi_avg > 0.4:
            high_prediction_cases.append({
                'core_id': core_id,
                'patient_id': patient_id,
                'cls_score': cls_score if cls_score is not None else 'N/A',
                'roi_avg': roi_avg,
                'gt_label': 'Cancer' if gt_label else 'Benign',
                'gt_involvement': gt_involvement,
                'type': 'ROI'
            })
        
        # Accumulate metrics for this batch
        evaluator(data)

    table = evaluator.accumulator.compute()
    table.to_csv(os.path.join(args.output_dir, "metrics_by_core.csv"))
    
    # Log high prediction cases
    if is_rl_model and len(high_prediction_cases) > 0:
        print(f"\n=== High Prediction Cases (CLS or ROI > 40%) ===")
        print(f"Found {len(high_prediction_cases)} cases with high predictions:")
        for case in high_prediction_cases[:20]:  # Show first 20
            cls_str = f"{case['cls_score']:.1%}" if isinstance(case['cls_score'], float) else str(case['cls_score'])
            roi_str = f"{case['roi_avg']:.1%}" if isinstance(case['roi_avg'], float) else str(case['roi_avg'])
            print(f"  {case['patient_id']}/{case['core_id']}: "
                  f"CLS={cls_str}, ROI={roi_str}, "
                  f"GT={case['gt_label']} (Inv: {case['gt_involvement']:.1%})")
        if len(high_prediction_cases) > 20:
            print(f"  ... and {len(high_prediction_cases) - 20} more")
        
        # Save to CSV
        import pandas as pd
        df_high_preds = pd.DataFrame(high_prediction_cases)
        df_high_preds.to_csv(
            os.path.join(args.output_dir, "high_prediction_cases.csv"), index=False
        )
        print(f"\nSaved high prediction cases to: {os.path.join(args.output_dir, 'high_prediction_cases.csv')}")

    metrics = evaluator.aggregate_metrics()
    metrics["infer_time"] = np.array(accumulator["infer_time"]).mean()
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Add RL-specific metrics
    if is_rl_model and rl_accumulator:
        print("\n=== RL Attention Statistics ===")
        # Only compute coordinate statistics if we have coordinates (discrete mode)
        if len(rl_accumulator['attention_coords']) > 0:
            all_coords = torch.cat(rl_accumulator['attention_coords'], dim=0)  # (N, k, 2)
            metrics['rl_attention_mean_x'] = float(all_coords[:, :, 0].mean())
            metrics['rl_attention_mean_y'] = float(all_coords[:, :, 1].mean())
            metrics['rl_attention_std_x'] = float(all_coords[:, :, 0].std())
            metrics['rl_attention_std_y'] = float(all_coords[:, :, 1].std())
            
            print(f"Average attention X: {metrics['rl_attention_mean_x']:.2f} ± {metrics['rl_attention_std_x']:.2f}")
            print(f"Average attention Y: {metrics['rl_attention_mean_y']:.2f} ± {metrics['rl_attention_std_y']:.2f}")
            
            # Save attention coordinates for further analysis
            np.save(
                os.path.join(args.output_dir, "rl_attention_coords.npy"),
                all_coords.numpy()
            )
        else:
            print("No attention coordinates available (continuous mode)")
            # For continuous mode, we still have attention maps
            if len(rl_accumulator['attention_maps']) > 0:
                print(f"Collected {len(rl_accumulator['attention_maps'])} attention maps")
        
        # === NEW: Comprehensive RL Evaluation Metrics ===
        if len(rl_accumulator.get('attention_maps', [])) > 0:
            print("\n=== RL Attention-Involvement Metrics ===")
            try:
                # Concatenate all attention maps
                all_attention = torch.cat(rl_accumulator['attention_maps'], dim=0)  # (N, H, W) or (N, 1, H, W)
                if all_attention.ndim == 4:
                    all_attention = all_attention.squeeze(1)
                all_attention_np = all_attention.numpy()
                
                # Get involvement and labels from the results table
                involvement_values = table['involvement'].values
                label_values = table['label'].values
                
                # Ensure same length
                N = min(len(all_attention_np), len(involvement_values))
                all_attention_np = all_attention_np[:N]
                involvement_values = involvement_values[:N]
                label_values = label_values[:N]
                
                # Mean attention per sample
                mean_attention = all_attention_np.mean(axis=(1, 2))
                
                # 1. Attention-Involvement Correlation
                from scipy.stats import spearmanr, pearsonr
                corr_spearman, p_value = spearmanr(mean_attention, involvement_values)
                metrics['rl/attention_involvement_correlation_spearman'] = float(corr_spearman)
                metrics['rl/attention_involvement_correlation_pvalue'] = float(p_value)
                print(f"Attention-Involvement Correlation (Spearman): {corr_spearman:.4f} (p={p_value:.4e})")
                
                corr_pearson, _ = pearsonr(mean_attention, involvement_values)
                metrics['rl/attention_involvement_correlation_pearson'] = float(corr_pearson)
                print(f"Attention-Involvement Correlation (Pearson): {corr_pearson:.4f}")
                
                # 2. Benign vs Cancer Attention
                benign_mask = label_values == 0
                cancer_mask = label_values == 1
                
                if benign_mask.sum() > 0:
                    benign_mean_attn = mean_attention[benign_mask].mean()
                    metrics['rl/benign_mean_attention'] = float(benign_mean_attn)
                    print(f"Benign Mean Attention: {benign_mean_attn:.4f}")
                    
                    # Sparsity: % below threshold
                    SPARSITY_THRESH = 0.1
                    benign_sparsity = (all_attention_np[benign_mask] < SPARSITY_THRESH).mean()
                    metrics['rl/benign_attention_sparsity'] = float(benign_sparsity)
                    print(f"Benign Attention Sparsity (<{SPARSITY_THRESH}): {benign_sparsity:.4f}")
                
                if cancer_mask.sum() > 0:
                    cancer_mean_attn = mean_attention[cancer_mask].mean()
                    metrics['rl/cancer_mean_attention'] = float(cancer_mean_attn)
                    print(f"Cancer Mean Attention: {cancer_mean_attn:.4f}")
                
                # Attention Contrast
                if benign_mask.sum() > 0 and cancer_mask.sum() > 0:
                    attention_contrast = cancer_mean_attn - benign_mean_attn
                    metrics['rl/attention_contrast'] = float(attention_contrast)
                    print(f"Attention Contrast (Cancer - Benign): {attention_contrast:.4f}")
                
                # 3. Attention AUROC
                if len(np.unique(label_values)) == 2:
                    from sklearn.metrics import roc_auc_score
                    attention_auroc = roc_auc_score(label_values, mean_attention)
                    metrics['rl/attention_auroc'] = float(attention_auroc)
                    print(f"Attention AUROC: {attention_auroc:.4f}")
                
                # 4. Calibration Error
                calibration_error = np.abs(mean_attention - involvement_values).mean()
                metrics['rl/attention_calibration_error'] = float(calibration_error)
                print(f"Attention Calibration Error: {calibration_error:.4f}")
                
                # 5. Attention by involvement bin
                for low, high in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
                    bin_mask = (involvement_values >= low) & (involvement_values < high)
                    if bin_mask.sum() > 0:
                        bin_mean_attn = mean_attention[bin_mask].mean()
                        metrics[f'rl/attention_at_involvement_{low:.1f}_{high:.1f}'] = float(bin_mean_attn)
                
                # 6. Attention Entropy (diversity)
                attention_flat = all_attention_np.reshape(N, -1)
                attention_probs = attention_flat / (attention_flat.sum(axis=1, keepdims=True) + 1e-8)
                entropy = -(attention_probs * np.log(attention_probs + 1e-8)).sum(axis=1)
                max_entropy = np.log(attention_probs.shape[1])
                normalized_entropy = entropy / max_entropy
                metrics['rl/attention_entropy_mean'] = float(normalized_entropy.mean())
                metrics['rl/attention_entropy_std'] = float(normalized_entropy.std())
                print(f"Attention Entropy: {normalized_entropy.mean():.4f} ± {normalized_entropy.std():.4f}")
                
                # 7. Needle Focus Ratio (KEY INTERPRETABILITY METRIC)
                # For cancer cases: is attention higher inside needle than outside?
                if 'needle_mask' in rl_accumulator and 'prostate_mask' in rl_accumulator:
                    needle_masks = rl_accumulator.get('needle_mask', [])
                    prostate_masks = rl_accumulator.get('prostate_mask', [])
                    if len(needle_masks) > 0:
                        needle_masks_np = torch.cat(needle_masks, dim=0).numpy()[:N]
                        prostate_masks_np = torch.cat(prostate_masks, dim=0).numpy()[:N]
                        
                        if needle_masks_np.ndim == 4:
                            needle_masks_np = needle_masks_np.squeeze(1)
                        if prostate_masks_np.ndim == 4:
                            prostate_masks_np = prostate_masks_np.squeeze(1)
                        
                        # Resize if needed
                        if needle_masks_np.shape[-2:] != all_attention_np.shape[-2:]:
                            import torch.nn.functional as F
                            needle_masks_np = F.interpolate(
                                torch.from_numpy(needle_masks_np).unsqueeze(1).float(),
                                size=all_attention_np.shape[-2:],
                                mode='nearest'
                            ).squeeze(1).numpy()
                            prostate_masks_np = F.interpolate(
                                torch.from_numpy(prostate_masks_np).unsqueeze(1).float(),
                                size=all_attention_np.shape[-2:],
                                mode='nearest'
                            ).squeeze(1).numpy()
                        
                        needle_focus_ratios = []
                        for i in range(N):
                            if label_values[i] == 0:  # Skip benign
                                continue
                            
                            attn_i = all_attention_np[i]
                            needle_i = needle_masks_np[i] > 0.5
                            prostate_i = prostate_masks_np[i] > 0.5
                            outside_needle = prostate_i & (~needle_i)
                            
                            if needle_i.sum() > 0 and outside_needle.sum() > 0:
                                mean_inside = attn_i[needle_i].mean()
                                mean_outside = attn_i[outside_needle].mean()
                                if mean_outside > 1e-6:
                                    ratio = mean_inside / mean_outside
                                    needle_focus_ratios.append(ratio)
                        
                        if len(needle_focus_ratios) > 0:
                            mean_ratio = float(np.mean(needle_focus_ratios))
                            correct_pct = float(np.mean([r > 1.0 for r in needle_focus_ratios]) * 100)
                            metrics['rl/cancer_needle_focus_ratio'] = mean_ratio
                            metrics['rl/cancer_needle_focus_correct_pct'] = correct_pct
                            print(f"Cancer Needle Focus Ratio: {mean_ratio:.4f}")
                            print(f"Cancer Needle Focus Correct %: {correct_pct:.1f}%")
                
            except Exception as e:
                print(f"Warning: Could not compute RL metrics: {e}")

        # Optionally save per-point attention probabilities and coordinates as CSV
        if 'rl_attention_point_records' in locals() and len(rl_attention_point_records) > 0:
            df_points = pd.DataFrame(rl_attention_point_records)
            df_points.to_csv(
                os.path.join(args.output_dir, "rl_attention_points.csv"), index=False
            )

    print("\n=== Test Metrics ===")
    print(json.dumps(metrics, indent=4))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def do_calibration_pixel_wise_balanced_bce(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # extract all pixel predictions from val loader
    pixel_preds, pixel_labels, core_ids = extract_all_pixel_predictions(
        model, loaders["val"], device
    )
    core_ids = np.array(core_ids)

    # fit temperature and bias to center and scale the predictions
    temp = nn.Parameter(torch.ones(1))
    bias = nn.Parameter(torch.zeros(1))

    from torch.optim import LBFGS

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(temp)

    optim = LBFGS(params, lr=1e-3, max_iter=100, line_search_fn="strong_wolfe")

    # weight the loss to account for class imbalance
    pos_weight = (1 - pixel_labels).sum() / pixel_labels.sum()
    # encourage sensitivity over specificity
    pos_weight *= 1.6

    def closure():
        optim.zero_grad()
        logits = pixel_preds / temp + bias
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits[:, 0], pixel_labels)
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(temp)
    model.bias.data.copy_(bias)


def do_calibration_bag_wise(
    model,
    loaders,
    calibrate_bias=True,
    calibrate_temperature=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    bags_of_logits, involvement, label = extract_all_bag_predictions(
        model, loaders["val"], device
    )

    # fit temperature and bias to center and scale the predictions
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device))

    from torch.optim import LBFGS

    pos_weight = (1 - label).sum() / label.sum()

    params = []
    if calibrate_bias:
        params.append(bias)
    if calibrate_temperature:
        params.append(log_temp)

    optim = LBFGS(params, lr=1e-1, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for bag_i, involvement_i, label_i in zip(bags_of_logits, involvement, label):
            bag_i = bag_i / log_temp.exp() + bias
            bag_i = bag_i.sigmoid()
            bag_i_mean = bag_i.mean()
            loss_i = (
                -involvement_i * bag_i_mean.log()
                - (1 - involvement_i) * (1 - bag_i_mean).log()
            )
            if label_i:
                loss_i = loss_i * pos_weight
            loss = loss + loss_i
        loss.backward()
        return loss

    for i in range(10):
        print(optim.step(closure))

    model.temperature.data.copy_(log_temp.exp())
    model.bias.data.copy_(bias)


@torch.no_grad()
def extract_all_bag_predictions(model, loader, device):

    bags_of_logits = []
    involvement = []
    label = []
    
    is_rl_model = hasattr(model, 'is_rl') and model.is_rl

    for data in tqdm(loader, f"Running model..."):
        if is_rl_model:
            data = model(data, deterministic=True)
        else:
            data = model(data)
        bags_of_logits.extend(
            get_bags_of_predictions(
                data["cancer_logits"], data["prostate_mask"], data["needle_mask"]
            )
        )
        involvement.append(data["involvement"].to(device))
        label.append(data["label"].to(device))

    involvement = torch.cat(involvement)
    label = torch.cat(label)

    return bags_of_logits, involvement, label


def extract_all_pixel_predictions(model, loader, device):
    pixel_labels = []
    pixel_preds = []
    core_ids = []

    model.eval()
    model.to(device)
    
    is_rl_model = hasattr(model, 'is_rl') and model.is_rl

    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            if is_rl_model:
                data = model(data, deterministic=True)
            else:
                data = model(data)

            prostate_mask = data["prostate_mask"].to(device)
            needle_mask = data["needle_mask"].to(device)
            heatmap_logits = data["cancer_logits"]
            label = data["label"]
            core_id = data["core_id"]

            # compute predictions
            masks = (prostate_mask > 0.5) & (needle_mask > 0.5)

            predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

            labels = torch.zeros(len(predictions), device=predictions.device)
            for i in range(len(predictions)):
                labels[i] = label[batch_idx[i]]
            pixel_preds.append(predictions.cpu())
            pixel_labels.append(labels.cpu())

            core_ids.extend(core_id[batch_idx[i]] for i in range(len(predictions)))

    pixel_preds = torch.cat(pixel_preds)
    pixel_labels = torch.cat(pixel_labels)

    return pixel_preds, pixel_labels, core_ids


if __name__ == "__main__":
    main()

