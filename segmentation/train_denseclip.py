print("--- Starting train_denseclip.py ---")

# ... rest of your imports
from tqdm import tqdm # Add import at the top
import argparse
import os
import os.path as osp
import time
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import subprocess # Needed for collect_env_info -> init_distributed_mode fix

# Explicit imports of model classes from denseclip.models
from denseclip import (
    CLIPResNet,
    CLIPTextEncoder,
    CLIPVisionTransformer,
    CLIPResNetWithAttention,
    CLIPTextContextEncoder,
    ContextDecoder,
    DenseCLIP # Add DenseCLIP here
)
from datasets.ade20k import ADE20KSegmentation  # Custom dataset class
# Correct import path for utils functions
from denseclip.utils import setup_logger, set_random_seed, collect_env_info, init_distributed

import sys
sys.path.append("./") # Add current directory to path

def parse_args():
    parser = argparse.ArgumentParser(description='Train DenseCLIP on ADE20K')
    parser.add_argument('config', help='Path to config file') # Positional
    parser.add_argument('--work-dir', help='Directory to save logs and models') # Optional
    parser.add_argument('--resume', help='Checkpoint to resume from') # Optional
    parser.add_argument('--load', help='Checkpoint to load weights from') # Optional
    parser.add_argument('--seed', type=int, default=42, help='Random seed') # Optional
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs') # Optional
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable deterministic training') # Optional
    parser.add_argument('--local_rank', type=int, default=0) # Optional
    args = parser.parse_args()
    return args

def cleanup():
    if dist.is_initialized(): # Only destroy if initialized
        dist.destroy_process_group()


def build_dataloader(cfg, rank=0, world_size=1):
    # Access dataset parameters correctly from the nested structure
    train_dataset = ADE20KSegmentation(
        root=cfg['data']['path'],
        split='train',
        crop_size=cfg['data']['crop_size'],
        # Make sure 'scale_range' is the key in your YAML for scale
        scale=cfg['data'].get('scale_range', cfg['data'].get('scale')), # Handle potential key name difference
        ignore_label=cfg['data']['ignore_label']
    )

    val_dataset = ADE20KSegmentation(
        root=cfg['data']['path'],
        split='val',
        crop_size=cfg['data']['crop_size'],
        scale=(1.0, 1.0),  # No scaling for validation
        ignore_label=cfg['data']['ignore_label']
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        # Access training parameters correctly
        batch_size=cfg['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['training']['workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Typically 1 for validation in segmentation
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg['training']['workers'],
        pin_memory=True
    )

    return train_loader, val_loader, ADE20KSegmentation.CLASSES


# Renamed to match the call in main() and mp.spawn
# Modified train_worker
def train_worker(rank, world_size, args, cfg, state_dict=None): # Add state_dict argument
    # Initialize distributed training if needed
    if world_size > 1:
        # Use the imported init_distributed function
        init_distributed(rank, world_size)

    # --- Setup Logging ---
    effective_work_dir = args.work_dir if args.work_dir else cfg.get('training', {}).get('work_dir', 'work_dirs/default')
    os.makedirs(effective_work_dir, exist_ok=True)
    logger = setup_logger(effective_work_dir, rank)
    logger.info(f'Running on rank {rank}')
    # Determine device based on rank and availability
    device = torch.device('cuda', rank) if world_size > 1 else \
             (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info(f"Using device: {device}")
    if rank == 0:
        logger.info(f'Config:\n{yaml.dump(cfg)}')
        # Log environment info
        try:
             env_info_str = collect_env_info(); logger.info(f"Environment Info:\n{env_info_str}")
        except Exception as e: logger.warning(f"Could not collect env info: {e}")


    # --- Set Random Seed ---
    set_random_seed(args.seed, deterministic=args.deterministic)

    # --- Build Model ---
    model_type = cfg['model']['type']
    model_cfg = cfg['model'].copy() # Get model config

    # Remove keys that are handled explicitly or not part of the model's __init__ signature
    model_cfg.pop('pretrained', None)
    model_cfg.pop('download_dir', None)
    model_cfg.pop('type', None)
    backbone_cfg = model_cfg.pop('backbone', None)
    text_encoder_cfg = model_cfg.pop('text_encoder', None)
    decode_head_cfg = model_cfg.pop('decode_head', None)
    context_decoder_cfg = model_cfg.pop('context_decoder', None)
    neck_cfg = model_cfg.pop('neck', None)
    explicit_context_length = model_cfg.pop('context_length', 5) # Default to 5 if not in top level

    if model_type == "DenseCLIP":
         # Build dataloader first temporarily to get class names
         # Note: This might load data unnecessarily if dataset is large.
         # Consider passing class names via config if possible.
         try:
             temp_loader, _, class_names_from_data = build_dataloader(cfg, rank, world_size)
             del temp_loader # Don't need the loader itself here
         except Exception as e:
             logger.error(f"Failed to build dataloader to get class names: {e}")
             # Provide a default or raise error if class names are strictly required
             class_names_from_data = [f'class_{i}' for i in range(cfg.get('data',{}).get('classes', 150))] # Example fallback
             logger.warning(f"Using default class names based on num_classes: {len(class_names_from_data)}")


         model = DenseCLIP(
             backbone=backbone_cfg, # Pass the extracted config dicts
             text_encoder=text_encoder_cfg,
             context_decoder=context_decoder_cfg,
             decode_head=decode_head_cfg,
             neck=neck_cfg,
             class_names=class_names_from_data, # Pass class names from dataset
             context_length=explicit_context_length, # Pass explicitly
             # Pass remaining model params from model_cfg using **
             **model_cfg
         )
    else:
         raise ValueError(f"Model type '{model_type}' not recognized or handled")

    model = model.to(device) # Move model to the correct device

    # --- Load Weights ---
    if args.load:
        try:
            checkpoint = torch.load(args.load, map_location=device)
            if 'state_dict' in checkpoint:
                weights_to_load = checkpoint['state_dict']
                if all(key.startswith('module.') for key in weights_to_load): weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}
                msg = model.load_state_dict(weights_to_load, strict=False); logger.info(f"Loaded --load ckpt: {args.load}");
                if rank == 0: logger.info(f"Load Msg: {msg}")
            else: logger.warning(f"--load ckpt missing 'state_dict'.")
        except FileNotFoundError: logger.error(f"--load checkpoint file not found: {args.load}")
        except Exception as e: logger.error(f"Error loading --load ckpt {args.load}: {e}")
    elif state_dict: # Load weights passed from main (pretrained weights)
         if all(key.startswith('module.') for key in state_dict): state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
         current_model_keys = set(model.state_dict().keys()); filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_model_keys}
         if not filtered_state_dict: logger.warning("Pretrained state_dict had no matching keys.")
         else: msg = model.load_state_dict(filtered_state_dict, strict=False); logger.info("Loaded pretrained weights.");
         if rank == 0: logger.info(f"Load Msg: {msg}")


    # --- Setup DDP ---
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # --- Build Actual Dataloaders ---
    train_loader, val_loader, _ = build_dataloader(cfg, rank, world_size)


    # --- Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=cfg['training']['min_lr'])

    # --- Loss Function ---
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg['data']['ignore_label'])

    # --- TensorBoard ---
    writer = None
    if rank == 0:
        tensorboard_log_dir = osp.join(effective_work_dir, 'tf_logs'); os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # --- Resume Logic ---
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
            # if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler']) # Optional

            if 'state_dict' in checkpoint:
                weights_to_load = checkpoint['state_dict']
                is_ddp_checkpoint = all(key.startswith('module.') for key in weights_to_load)
                current_model_is_ddp = isinstance(model, DDP)

                if current_model_is_ddp and not is_ddp_checkpoint: weights_to_load = {'module.' + k: v for k,v in weights_to_load.items()}
                elif not current_model_is_ddp and is_ddp_checkpoint: weights_to_load = {k.replace('module.', '', 1): v for k, v in weights_to_load.items()}

                current_model_state = model.state_dict()
                filtered_resume_weights = {k: v for k, v in weights_to_load.items() if k in current_model_state and v.shape == current_model_state[k].shape}
                current_model_state.update(filtered_resume_weights)
                msg = model.load_state_dict(current_model_state, strict=False)

                logger.info(f"Resuming training from epoch {start_epoch} using checkpoint: {args.resume}")
                if rank == 0: logger.info(f"Resume Load State Dict Message: {msg}")
            else:
                 logger.warning(f"Resume checkpoint '{args.resume}' missing 'state_dict'.")

        except FileNotFoundError: logger.error(f"Resume checkpoint not found: {args.resume}. Starting fresh.")
        except Exception as e: logger.error(f"Error loading resume checkpoint {args.resume}: {e}. Starting fresh.")


    # --- Training Loop ---
    total_epochs = cfg['training']['epochs']
    # log_interval = cfg['training']['log_interval'] # No longer needed for tqdm bar
    eval_interval = cfg['training'].get('eval_interval', 1)
    save_interval = cfg['training'].get('save_interval', 1)

    logger.info(f"Starting Training from Epoch {start_epoch}...")
    for epoch in range(start_epoch, total_epochs):
        model.train() # Set model to training mode
        if world_size > 1: train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = len(train_loader)

        # --- Use tqdm for progress bar on rank 0 ---
        pbar = None
        if rank == 0:
            pbar = tqdm(total=num_batches, desc=f"Epoch {epoch}/{total_epochs-1} Train", unit="batch")

        for i, batch_data in enumerate(train_loader):
             try: # Add try-except around batch processing
                 if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: images, targets = batch_data
                 elif isinstance(batch_data, dict): images, targets = batch_data.get('img'), batch_data.get('gt_semantic_seg')
                 else: logger.error(f"Unexpected batch format: {type(batch_data)}"); continue
                 if images is None or targets is None: logger.error(f"Missing img/gt_semantic_seg"); continue

                 images = images.to(device, non_blocking=True)
                 targets = targets.to(device, non_blocking=True).long()

                 optimizer.zero_grad()

                 # --- Forward Pass & Loss ---
                 # Model forward pass in training mode (return_loss=True) returns a dictionary
                 output_dict = model(images, return_loss=True, gt_semantic_seg=targets, img_metas=None)

                 # --- START CHANGE: Loss Calculation from Dict ---
                 if not isinstance(output_dict, dict) or 'main_output' not in output_dict:
                      logger.error(f"Model forward pass did not return expected dictionary with 'main_output'. Got: {type(output_dict)}")
                      # Skip batch or raise error depending on desired behavior
                      continue # Skip this batch

                 # Get main logits
                 main_logits = output_dict['main_output']

                 # Ensure logits and targets have compatible shapes for loss calculation
                 # Check spatial dimensions - resize logits if necessary
                 gt_h, gt_w = targets.shape[-2:]
                 if main_logits.shape[-2:] != (gt_h, gt_w):
                        # Use align_corners setting from the model/head if available
                        align_corners = getattr(model.module if isinstance(model, DDP) else model, 'align_corners', False)
                        main_logits_resized = F.interpolate(
                             main_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=align_corners
                        )
                        # logger.debug(f"Resized main_logits from {main_logits.shape} to {main_logits_resized.shape} for loss.") # Optional debug
                 else:
                        main_logits_resized = main_logits

                 # Calculate the primary loss using the main output logits
                 try:
                     loss = criterion(main_logits_resized, targets)
                 except Exception as e:
                      logger.error(f"Error calculating main loss: {e}")
                      logger.error(f"Logits shape: {main_logits_resized.shape}, Targets shape: {targets.shape}, Targets dtype: {targets.dtype}")
                      # Consider adding more debug info like min/max target values:
                      # logger.error(f"Target min: {targets.min()}, Target max: {targets.max()}, Unique targets: {torch.unique(targets)}")
                      continue # Skip batch on loss error


                 # --- Optional: Add auxiliary losses if calculated by model ---
                 if 'aux_losses' in output_dict and isinstance(output_dict['aux_losses'], dict):
                       aux_losses_sum = torch.tensor(0.0, device=loss.device) # Initialize sum on correct device
                       for loss_name, aux_loss_val in output_dict['aux_losses'].items():
                            if torch.is_tensor(aux_loss_val) and aux_loss_val.requires_grad: # Check if it's a valid loss tensor
                                logger.debug(f"Adding aux loss: {loss_name} = {aux_loss_val.item():.4f}")
                                # Add aux losses (consider weighting factors from config if needed)
                                # Example: weight = cfg['model'].get('aux_loss_weights', {}).get(loss_name, 1.0)
                                # aux_losses_sum = aux_losses_sum + weight * aux_loss_val
                                aux_losses_sum = aux_losses_sum + aux_loss_val
                            elif torch.is_tensor(aux_loss_val):
                                 logger.warning(f"Aux loss '{loss_name}' does not require grad, skipping.")
                            else:
                                logger.warning(f"Non-tensor or invalid value found in aux_losses dict: {loss_name}")
                       loss = loss + aux_losses_sum # Add summed aux losses to main loss
                 # --- End Optional: Add auxiliary losses ---

                 # --- END CHANGE ---


                 # --- Backward and Optimize ---
                 if torch.isnan(loss) or torch.isinf(loss):
                      logger.error(f"NaN or Inf loss detected: {loss.item()}. Skipping backward/step.")
                      continue # Skip batch

                 try:
                     loss.backward()
                     # Optional: Gradient clipping
                     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     optimizer.step()
                 except RuntimeError as e:
                      logger.error(f"Runtime error during backward or step: {e}")
                      optimizer.zero_grad() # Clear gradients if step failed
                      continue # Skip batch


                 batch_loss = loss.item() # Get scalar value of total loss
                 epoch_loss += batch_loss

                 # --- Update tqdm postfix on rank 0 ---
                 if pbar: pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

             except Exception as batch_e:
                  logger.error(f"Error in training batch {i}: {batch_e}", exc_info=True)

             finally:
                  # Ensure progress bar updates
                  if pbar: pbar.update(1)


        # Close tqdm progress bar
        if pbar: pbar.close()

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        if rank == 0:
             logger.info(f"--- Epoch {epoch}/{total_epochs-1} Finished --- Avg Train Loss: {avg_epoch_loss:.4f} ---")
             if writer: writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)

        scheduler.step()

        # --- Validation ---
        if rank == 0 and (epoch + 1) % eval_interval == 0:
             validate(model, val_loader, criterion, epoch, writer, logger, device, effective_work_dir) # Pass device and work_dir

        # --- Save Checkpoint ---
        if rank == 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = osp.join(effective_work_dir, 'checkpoints', f'epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # --- Final Cleanup ---
    if writer and rank == 0: writer.close()
    if world_size > 1: cleanup()

def validate(model, val_loader, criterion, epoch, writer, logger, device, work_dir): # Added work_dir back for consistency
    # --- START CHANGE: Improved Rank/Device Check ---
    # Check if it's rank 0 in distributed, OR if it's single process on CPU, OR single process on default CUDA ('cuda' or 'cuda:0')
    is_primary_process = False # Default to False
    if dist.is_initialized(): # Check if distributed environment is initialized
        if dist.get_rank() == 0:
            is_primary_process = True
    else: # Not distributed
        if str(device) == 'cpu' or str(device).startswith('cuda'): # Check for CPU or any CUDA device in single process mode
             is_primary_process = True

    print(f"DEBUG: Entering validate function for epoch {epoch} on device {device}. Is primary process for logging/pbar: {is_primary_process}") # Updated DEBUG
    # --- END CHANGE ---
    logger.info(f"--- Starting Validation Epoch: {epoch} ---")

    # Get the underlying model if wrapped in DDP
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval() # Set model to evaluation mode

    total_loss = 0
    num_batches = len(val_loader)

    print(f"DEBUG: Validation - Num Batches: {num_batches}")
    if num_batches == 0:
        logger.warning("Validation loader has zero batches. Skipping validation loop.")
        print("DEBUG: Exiting validate function early (0 batches).")
        return # Exit validate function early if no batches

    val_pbar = None
    # --- CHANGE: Use updated check ---
    if is_primary_process:
        print(f"DEBUG: Creating validation progress bar for {num_batches} batches.")
        val_pbar = tqdm(total=num_batches, desc=f"Epoch {epoch} Validate", unit="batch", leave=False) # leave=False hides bar after completion

    with torch.no_grad():
        print(f"DEBUG: Starting validation loop for epoch {epoch}.")
        for i, batch_data in enumerate(val_loader):
            # --- CHANGE: Only print debug periodically on primary process ---
            if i % 50 == 0 and is_primary_process:
                 print(f"DEBUG: Processing validation batch {i}/{num_batches}")
            try: # Add try-except for validation batch
                # --- Data Handling ---
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2: images, targets = batch_data
                elif isinstance(batch_data, dict): images, targets = batch_data.get('img'), batch_data.get('gt_semantic_seg')
                else: logger.error(f"Val batch format error"); continue
                if images is None or targets is None: logger.error(f"Val missing img/gt"); continue

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long()

                # --- Forward Pass ---
                outputs = model_to_eval(images, return_loss=False) # Get logits directly

                # --- Extract Logits ---
                if isinstance(outputs, dict): logits = outputs.get('main_output', outputs.get('logits'))
                elif torch.is_tensor(outputs): logits = outputs
                else: logger.error(f"Unexpected val output type: {type(outputs)}"); continue
                if logits is None: logger.error("Could not get logits from val output"); continue

                # --- Calculate Loss (Optional but useful for basic check) ---
                gt_h, gt_w = targets.shape[-2:]
                if logits.shape[-2:] != (gt_h, gt_w):
                     # Use align_corners from the model if available
                     align_corners = getattr(model_to_eval, 'align_corners', False)
                     logits_resized = F.interpolate(logits, size=(gt_h, gt_w), mode='bilinear', align_corners=align_corners)
                else: logits_resized = logits

                loss = criterion(logits_resized, targets)
                if not (torch.isnan(loss) or torch.isinf(loss)): # Check loss is valid before adding
                      total_loss += loss.item()
                else:
                      logger.warning(f"NaN or Inf loss detected during validation batch {i}. Skipping loss accumulation for this batch.")

                # --- Metrics Calculation and Image Saving are SKIPPED for this version ---

            except Exception as batch_e:
                 logger.error(f"Error in validation batch {i}: {batch_e}", exc_info=True)

            finally:
                 # --- Update Progress Bar ---
                 if val_pbar:
                     val_pbar.update(1)
                     # Update postfix less often to avoid excessive updates
                     if i % 10 == 0 or i == num_batches - 1:
                          avg_loss_so_far = total_loss / (i + 1) if (i + 1) > 0 else 0.0
                          val_pbar.set_postfix(avg_loss=f"{avg_loss_so_far:.4f}")


    # Close validation progress bar
    if val_pbar: val_pbar.close()

    # --- ADD PRINT ---
    print(f"DEBUG: Finished validation loop for epoch {epoch}.")

    # --- Log Simplified Metrics (on primary process) ---
    # --- CHANGE: Use updated check ---
    if is_primary_process:
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        # This log message should now appear
        logger.info(f'--- Validation Epoch: {epoch} --- Avg Loss: {avg_loss:.4f} --- (Detailed metrics skipped) ---')
        if writer:
            writer.add_scalar('val/epoch_loss', avg_loss, epoch)

    # --- ADD PRINT ---
    print(f"DEBUG: Exiting validate function for epoch {epoch}.")

def save_checkpoint(model, optimizer, epoch, path): # Pass path directly
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    state = {
        'epoch': epoch,
        'state_dict': model_state,
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(), # Optional: save scheduler state
    }
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(state, path)


# Main execution block
if __name__ == '__main__':
    args = parse_args()
    try:
        with open(args.config) as f: cfg = yaml.safe_load(f)
    except Exception as e: print(f"Error loading config: {e}"); sys.exit(1)

    if args.work_dir is None:
        config_name = osp.splitext(osp.basename(args.config))[0]
        args.work_dir = osp.join('work_dirs', config_name)
    os.makedirs(args.work_dir, exist_ok=True)

    try: # Save config used
        with open(osp.join(args.work_dir, 'used_config.yaml'), 'w') as f: yaml.dump(cfg, f, default_flow_style=False)
    except Exception as e: print(f"Warning: Could not save config: {e}")

    # --- Pre-download/Load Weights Logic (mostly unchanged, ensure ensure_weights is defined) ---
    pretrained_path = None
    state_dict_to_pass = None
    if cfg.get('model', {}).get('pretrained'):
        if args.gpus <= 1 or args.local_rank == 0:
            pretrained_url_or_path = cfg['model']['pretrained']
            save_dir = cfg['model'].get('download_dir', 'pretrained')

            def ensure_weights(url_or_path, save_dir): # Define ensure_weights here or import
                os.makedirs(save_dir, exist_ok=True)
                if str(url_or_path).startswith(('http:', 'https:')):
                    try: import wget
                    except ImportError: print("Error: wget not installed."); return None
                    filename = url_or_path.split('/')[-1]
                    save_path = osp.join(save_dir, filename)
                    if not osp.exists(save_path):
                        print(f"Rank {args.local_rank}: Downloading {filename}...")
                        try: wget.download(url_or_path, out=save_path); print(f"\nDownload complete!")
                        except Exception as e: print(f"\nError downloading {url_or_path}: {e}"); return None
                    else: print(f"Rank {args.local_rank}: Weights found at {save_path}")
                    return save_path
                else:
                    local_path = url_or_path
                    if not osp.isabs(local_path):
                         config_dir = osp.dirname(args.config)
                         potential_path = osp.join(config_dir, local_path)
                         if osp.exists(potential_path): local_path = potential_path
                         else: local_path = osp.join(osp.dirname(__file__) if "__file__" in locals() else ".", local_path)
                    if osp.exists(local_path): print(f"Rank {args.local_rank}: Using local weights: {local_path}"); return local_path
                    else: print(f"Rank {args.local_rank}: Local weights not found: {local_path}"); return None

            pretrained_path = ensure_weights(pretrained_url_or_path, save_dir)
            if pretrained_path:
                try: state_dict_to_pass = torch.load(pretrained_path, map_location='cpu'); print(f"Loaded state dict from {pretrained_path}")
                except Exception as e: print(f"Error loading state dict {pretrained_path}: {e}"); state_dict_to_pass = None
            else: state_dict_to_pass = None

    if args.gpus > 1 and cfg.get('model', {}).get('pretrained'):
        if not dist.is_initialized(): init_distributed(args.local_rank, args.gpus)
        print(f"Rank {args.local_rank}: Barrier for weights..."); dist.barrier(); print(f"Rank {args.local_rank}: Passed barrier.")

    # --- Launch Training ---
    world_size = args.gpus
    if world_size > 1:
        print(f"Spawning {world_size} processes.")
        mp.spawn(train_worker, args=(world_size, args, cfg, state_dict_to_pass), nprocs=world_size, join=True)
    else:
        print("Running in single process mode.")
        train_worker(0, 1, args, cfg, state_dict_to_pass)
