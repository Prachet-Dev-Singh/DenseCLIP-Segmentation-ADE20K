import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Use standard logging
import numpy as np

# Explicitly import necessary components from within the denseclip package
from .models import (
    CLIPResNet, CLIPTextEncoder, CLIPVisionTransformer,
    CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder
)
# Import head classes if they are defined locally (adjust path if needed)
# Assuming FPNHead and IdentityHead might need custom implementations or replacements
# from .heads import IdentityHead # Uncomment if IdentityHead is defined and needed
# Need to handle FPN and FPNHead - replace with standard implementations if possible
# Example using torchvision FPN (might need adaptation)
try:
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
    from torchvision.models.segmentation.fcn import FCNHead # Example replacement for FPNHead
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    FeaturePyramidNetwork = None
    FCNHead = None
    print("Warning: torchvision not found or FPN/FCNHead not available. Neck and Decode Head need replacement.")


# --- START CHANGE ---
# Import tokenize from the utils module
from .utils import tokenize
# --- END CHANGE ---

# Setup logger for this module
logger = logging.getLogger(__name__)

# ================== REPLACEMENTS FOR MMSEG/MMCV ================== #
class Registry:
    """Replacement for mmseg.models.builder.HEADS/SEGMENTORS"""
    def __init__(self):
        self._registry = {}
    
    def register_module(self, name=None):
        def decorator(module_class):
            module_name = name if name is not None else module_class.__name__
            self._registry[module_name] = module_class
            return module_class
        return decorator
    
    def build(self, cfg, **kwargs):
        if isinstance(cfg, dict):
            obj_type = cfg.pop('type')
            return self._registry[obj_type](**cfg, **kwargs)
        return self._registry[cfg](**kwargs)

SEGMENTORS = Registry()

def resize(input, size=None, scale_factor=None, mode='bilinear', 
           align_corners=None, warning=True):
    """Replacement for mmseg.ops.resize"""
    if warning and size is not None and align_corners:
        input_h, input_w = input.shape[2:]
        output_h, output_w = size
        if output_h > input_h or output_w > input_w:
            print(f"Warning: align_corners={align_corners} is recommended for downsizing")
    return F.interpolate(input, size=size, scale_factor=scale_factor, 
                        mode=mode, align_corners=align_corners)

def add_prefix(input_dict, prefix):
    """Replacement for mmseg.core.add_prefix"""
    return {f'{prefix}.{k}': v for k, v in input_dict.items()}

class BaseSegmentor(nn.Module):
    """Simplified replacement for mmseg.models.segmentors.base.BaseSegmentor"""
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
        
    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise NotImplementedError("Custom init_cfg not supported in this replacement")

# ================== ORIGINAL DENSECLIP CLASS (UNMODIFIED) ================== #

# Setup logger for this module
logger = logging.getLogger(__name__)

class DenseCLIP(nn.Module): # Inherit directly from nn.Module
    """
    DenseCLIP segmentor implementation without mmsegmentation dependencies.
    """
    def __init__(self,
                 backbone, # Config dict for backbone
                 text_encoder, # Config dict for text encoder
                 decode_head, # Config dict for decode head
                 class_names, # List of class names
                 context_length,
                 context_decoder=None, # Optional config dict
                 neck=None, # Optional config dict
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False, # Whether to feed text embeddings to decode head
                 tau=0.07,
                 auxiliary_head=None, # Optional config dict
                 identity_head=None, # Optional config dict
                 train_cfg=None, # Keep for potential future use, but not directly used by nn.Module
                 test_cfg=None, # Keep for potential future use, but not directly used by nn.Module
                 # pretrained is handled by the caller script loading state_dict
                 # init_cfg is an mmcv concept, removed
                 token_embed_dim=512, # Usually related to text token embedding before transformer
                 text_dim=1024, # Target dimension for text features after projection/encoder
                 **kwargs): # Use kwargs for flexibility
        super().__init__() # Call nn.Module's init

        self.class_names = class_names
        self.context_length = context_length
        self.context_feature = context_feature
        self.score_concat_index = score_concat_index
        self.text_head = text_head
        self.tau = tau
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg # Store test_cfg for potential use in inference logic
        self.num_classes = len(class_names) # Initialize num_classes

        # --- Direct Instantiation of Components ---

        # Backbone
        backbone_cfg = backbone.copy()
        backbone_type = backbone_cfg.pop('type')
        logger.info(f"Building backbone: {backbone_type}")
        resnet_layers = backbone_cfg.pop('layers', None)
        if backbone_type in ["CLIPResNet", "CLIPResNetWithAttention"] and resnet_layers is None:
             raise ValueError(f"'layers' argument is required for backbone type {backbone_type}")

        if backbone_type == "CLIPResNet":
             self.backbone = CLIPResNet(resnet_layers, **backbone_cfg)
             # --- Infer backbone output channels ---
             # For ResNet, the output channels of layer4 is typically width * 8 * expansion (4)
             backbone_out_channels = backbone_cfg.get('width', 64) * 8 * 4 # 2048 for width=64
        elif backbone_type == "CLIPResNetWithAttention":
             self.backbone = CLIPResNetWithAttention(resnet_layers, **backbone_cfg)
             # Output dim of AttentionPool layer determines the visual feature dim
             backbone_out_channels = backbone_cfg.get('output_dim', 1024) # Get from config or default
        elif backbone_type == "CLIPVisionTransformer":
              self.backbone = CLIPVisionTransformer(**backbone_cfg)
              # ViT output dim depends on its configuration (width or output_dim if projection exists)
              backbone_out_channels = backbone_cfg.get('output_dim', 512) # Check ViT config
        else:
             raise ValueError(f"Unsupported backbone type: {backbone_type}")
        logger.info(f"Inferred backbone output channels/dim: {backbone_out_channels}")


        # Text Encoder
        text_encoder_cfg = text_encoder.copy()
        text_encoder_type = text_encoder_cfg.pop('type')
        logger.info(f"Building text encoder: {text_encoder_type}")
        text_encoder_cfg.setdefault('context_length', 77)
        text_encoder_cfg.setdefault('vocab_size', 49408)
        # Infer text encoder output dimension (should match target text_dim)
        text_encoder_out_dim = text_encoder_cfg.get('embed_dim', text_dim) # Use embed_dim if specified, else default to text_dim
        if text_encoder_out_dim != text_dim:
             logger.warning(f"text_encoder embed_dim ({text_encoder_out_dim}) does not match model text_dim ({text_dim}). Alignment might be needed.")
             # Potentially override text_dim based on encoder config if that's intended
             # text_dim = text_encoder_out_dim

        if text_encoder_type == "CLIPTextEncoder":
             self.text_encoder = CLIPTextEncoder(**text_encoder_cfg)
        elif text_encoder_type == "CLIPTextContextEncoder":
              text_encoder_cfg.setdefault('context_length', 13)
              self.text_encoder = CLIPTextContextEncoder(**text_encoder_cfg)
        else:
             raise ValueError(f"Unsupported text_encoder type: {text_encoder_type}")

        # --- Add Visual Projection Layer IF needed ---
        self.vis_proj = None
        if backbone_out_channels != text_dim:
            logger.info(f"Visual feature dim ({backbone_out_channels}) != Text feature dim ({text_dim}). Adding projection layer.")
            self.vis_proj = nn.Conv2d(backbone_out_channels, text_dim, kernel_size=1)
        # --- End Add Visual Projection ---


        # Context Decoder (Optional)
        self.context_decoder = None
        if context_decoder:
            context_decoder_cfg = context_decoder.copy()
            context_decoder_type = context_decoder_cfg.pop('type')
            logger.info(f"Building context decoder: {context_decoder_type}")
            if context_decoder_type == "ContextDecoder":
                 # Ensure visual_dim matches the dimension *after* potential projection
                 context_decoder_cfg.setdefault('visual_dim', text_dim) # Assume it works with aligned dim
                 self.context_decoder = ContextDecoder(**context_decoder_cfg)
            else:
                 raise ValueError(f"Unsupported context_decoder type: {context_decoder_type}")

        # Neck (Optional)
        self.neck = None
        self._neck_out_keys = None
        if neck:
             # ... (neck instantiation logic as before) ...
             neck_cfg = neck.copy()
             neck_type = neck_cfg.pop('type')
             logger.info(f"Building neck: {neck_type}")
             if neck_type == "FPN" and FeaturePyramidNetwork is not None:
                 in_channels_list = neck_cfg.get('in_channels', [256, 512, 1024, 2048]) # Default, might need adjustment based on actual backbone output stages if vis_proj is used early
                 out_channels = neck_cfg.get('out_channels', 256)
                 num_outs = neck_cfg.get('num_outs', len(in_channels_list))
                 extra_blocks = None
                 if num_outs > len(in_channels_list): extra_blocks = LastLevelMaxPool()

                 self.neck = FeaturePyramidNetwork(
                      in_channels_list=in_channels_list,
                      out_channels=out_channels,
                      extra_blocks=extra_blocks
                 )
                 # Dynamically get output keys (requires dummy forward)
                 try:
                    dummy_input_neck = {str(i):torch.rand(1,c,64//(2**i),64//(2**i)) for i,c in enumerate(in_channels_list)}
                    self._neck_out_keys = list(self.neck(dummy_input_neck).keys())
                    logger.info(f"Using torchvision FPN. Output keys: {self._neck_out_keys}")
                 except Exception as e:
                     logger.warning(f"Could not dynamically determine FPN output keys: {e}. Assuming default ['0', '1', '2', '3', 'pool'] if extra_blocks exist.")
                     self._neck_out_keys = [str(i) for i in range(num_outs)] # Fallback naming
             elif neck_type == "FPN": logger.error("Torchvision FPN not available. Neck cannot be built.")
             else: raise ValueError(f"Unsupported neck type: {neck_type}")


        # Decode Head
        self.decode_head = None
        self._decode_head_cfg = None
        self.align_corners = False
        if decode_head:
            # ... (decode head instantiation logic as before, using FCNHead and replacing classifier) ...
            decode_head_cfg = decode_head.copy()
            decode_head_type = decode_head_cfg.pop('type')
            logger.info(f"Building decode head: {decode_head_type}")
            self._decode_head_cfg = decode_head_cfg # Store config
            self.align_corners = decode_head_cfg.get('align_corners', False)
            decode_num_classes = decode_head_cfg.get('num_classes', self.num_classes)
            if decode_num_classes != self.num_classes:
                 logger.warning(f"Decode head num_classes ({decode_num_classes}) != class_names length ({self.num_classes}). Using decode_head num_classes.")
                 self.num_classes = decode_num_classes

            if decode_head_type == "FPNHead" and FCNHead is not None:
                 in_channels = decode_head_cfg.get('in_channels', 256) # Should match neck out_channels
                 channels = decode_head_cfg.get('channels', 256)
                 self.decode_head = FCNHead(in_channels=in_channels, channels=channels)
                 num_intermediate_channels = channels
                 self.decode_head.classifier = nn.Conv2d(num_intermediate_channels, self.num_classes, kernel_size=1)
                 logger.info(f"Replaced FCNHead classifier for {self.num_classes} classes.")

            elif decode_head_type == "FPNHead":
                 logger.error("Torchvision FCNHead not available. Decode head cannot be built.")
            else:
                 try: # Handle custom heads
                      if decode_head_type == "IdentityHead":
                           from .heads import IdentityHead
                           self.decode_head = IdentityHead(**decode_head_cfg)
                      else: raise ValueError(f"Unsupported decode_head type: {decode_head_type}")
                 except ImportError: raise ValueError(f"Custom decode_head type '{decode_head_type}' specified but not found.")


        self.with_decode_head = self.decode_head is not None
        if not hasattr(self, 'num_classes') or self.num_classes is None:
             raise ValueError("Could not determine number of classes.")
        elif not self.with_decode_head:
             logger.warning("No decode head built.")


        # Auxiliary Head (Optional)
        self.auxiliary_head = None
        self.with_auxiliary_head = False
        if auxiliary_head:
             logger.warning("Auxiliary head instantiation not fully implemented.")
             self.with_auxiliary_head = True

        # Identity Head (Optional)
        self.identity_head = None
        self.with_identity_head = False
        if identity_head:
            logger.warning("Identity head instantiation logic needs refinement.")
            self.with_identity_head = True
            try:
                 from .heads import IdentityHead
                 id_head_cfg = identity_head.copy()
                 id_head_type = id_head_cfg.pop('type')
                 self.identity_head = IdentityHead(**id_head_cfg)
            except ImportError: logger.error("IdentityHead specified but not found.") ; self.with_identity_head = False
            except Exception as e: logger.error(f"Error instantiating IdentityHead: {e}") ; self.with_identity_head = False


        # --- Tokenization and Learnable Parameters ---
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in self.class_names])

        text_encoder_context_length = getattr(self.text_encoder, 'context_length', 77)
        prompt_context_length = text_encoder_context_length - self.context_length
        if prompt_context_length < 0:
             logger.warning(f"context_length ({self.context_length}) > text_encoder capacity ({text_encoder_context_length}). Prompt context length set to 0.")
             prompt_context_length = 0

        if isinstance(self.text_encoder, CLIPTextContextEncoder):
            _token_embed_dim = kwargs.get('token_embed_dim', 512)
            _text_dim_gamma = text_dim # Use the target text_dim for gamma
            self.contexts = nn.Parameter(torch.randn(1, prompt_context_length, _token_embed_dim))
            nn.init.trunc_normal_(self.contexts)
            self.gamma = nn.Parameter(torch.ones(_text_dim_gamma) * 1e-4)
            logger.info("Initialized learnable text contexts and gamma.")
        else:
            self.contexts = None
            self.gamma = None
            logger.info("Standard text encoder used. Learnable text contexts/gamma not initialized.")


        # --- Weight Initialization ---
        logger.info("Initializing weights for newly added components (if any)...")
        if self.vis_proj is not None: # Initialize the projection layer
            nn.init.kaiming_normal_(self.vis_proj.weight, mode='fan_out', nonlinearity='relu')
            if self.vis_proj.bias is not None: nn.init.constant_(self.vis_proj.bias, 0)
        # Add initialization for neck, decode_head etc. if needed
        # Example: if self.decode_head: self.decode_head.apply(self._init_weights_fn)

    def extract_feat(self, img):
        """Extract features from images using the backbone."""
        x = self.backbone(img)
        return x

    def _process_features(self, x):
        """
        Handles feature processing after backbone extraction...
        """
        _x_orig = list(x)
        visual_embeddings = None
        global_feat = None

        # Unpack backbone features
        if isinstance(x[-1], (list, tuple)) and len(x[-1]) == 2: # AttentionPool2d output
            logger.debug("Using features from AttentionPool2d.")
            _x_orig = list(x[:-1])
            global_feat, visual_embeddings = x[-1]
        else: # Standard backbone output (list/tuple of feature maps)
            #logger.warning("Backbone output not from AttentionPool2d. Using last feature map.")
            logger.debug("Backbone output not from AttentionPool2d. Using last feature map.") # Changed to debug
            visual_embeddings = x[-1]
            if visual_embeddings.dim() != 4:
                 raise ValueError(f"Expected last backbone feature map 4D, got {visual_embeddings.dim()}")
            global_feat = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).squeeze(-1).squeeze(-1)

        if visual_embeddings is None: raise ValueError("Could not determine visual_embeddings.")
        if global_feat is None: raise ValueError("Could not determine global_feat.")

        # Get initial visual dimensions before potential projection
        B, C_vis_orig, H_vis, W_vis = visual_embeddings.shape

        # --- Apply Visual Projection IF defined in __init__ ---
        if self.vis_proj is not None:
             logger.debug(f"Applying visual projection: {C_vis_orig} -> {self.vis_proj.out_channels}")
             visual_embeddings = self.vis_proj(visual_embeddings)
             # Update visual dimensions AFTER projection
             B, C_vis, H_vis, W_vis = visual_embeddings.shape # C_vis is now the projected dimension
             logger.debug(f"Shape after projection: {visual_embeddings.shape}")
        else:
             C_vis = C_vis_orig # Use original channel dim if no projection needed

        # Prepare visual context (using potentially projected features)
        if self.context_feature == 'attention':
             # Context typically uses the *projected* dimension if vis_proj exists
             if global_feat.dim() != 2 or global_feat.shape[0] != B or global_feat.shape[1] != C_vis: # Check against projected C_vis
                  # logger.warning(f"global_feat shape ({global_feat.shape}) or C_vis ({C_vis}) mismatch. Re-pooling projected features.")
                  logger.debug(f"global_feat shape ({global_feat.shape}) or C_vis ({C_vis}) mismatch. Re-pooling projected features.") # Changed to debug
                  # Recalculate global_feat from *projected* visual_embeddings
                  global_feat_ctx = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).view(B, C_vis)
             else:
                 # If original global_feat matches projected C_vis, use it, otherwise re-pool projected
                 # This assumes global_feat should align with the projected dim for context decoder
                 if global_feat.shape[1] == C_vis:
                    global_feat_ctx = global_feat
                 else:
                     logger.warning(f"Original global_feat dim ({global_feat.shape[1]}) != projected C_vis ({C_vis}). Re-pooling projected.")
                     global_feat_ctx = F.adaptive_avg_pool2d(visual_embeddings, (1, 1)).view(B, C_vis)

             visual_context = torch.cat([global_feat_ctx.unsqueeze(2), visual_embeddings.reshape(B, C_vis, H_vis*W_vis)], dim=2).permute(0, 2, 1)

        elif self.context_feature == 'backbone':
             last_backbone_feat = _x_orig[-1] # Use original last backbone feat for this context type
             B_b, C_b, H_b, W_b = last_backbone_feat.shape
             visual_context = last_backbone_feat.view(B_b, C_b, -1).permute(0, 2, 1)
             # Add projection here if context_decoder expects different dim than C_b
        else:
             raise ValueError(f"Invalid context_feature type: {self.context_feature}")
        logger.debug(f"Visual context shape ({self.context_feature}): {visual_context.shape}")

        # --- Text Feature Calculation ---
        if not hasattr(self, 'text_encoder') or self.text_encoder is None: raise AttributeError("text_encoder missing")
        text_embeddings_device = next(self.text_encoder.parameters()).device

        if isinstance(self.text_encoder, CLIPTextContextEncoder):
             if self.contexts is None: raise AttributeError("'contexts' parameter missing")
             text_embeddings = self.text_encoder(self.texts.to(text_embeddings_device), self.contexts.to(text_embeddings_device)).expand(B, -1, -1)
             if self.context_decoder:
                 if self.gamma is None: raise AttributeError("'gamma' parameter missing")
                 # Ensure visual_context is on correct device and potentially projected if decoder expects it
                 # visual_context_for_decoder = self.context_proj(visual_context) if needed else visual_context
                 text_diff = self.context_decoder(text_embeddings, visual_context.to(text_embeddings_device))
                 text_embeddings = text_embeddings + self.gamma.to(text_embeddings_device) * text_diff
        elif isinstance(self.text_encoder, CLIPTextEncoder):
             text_embeddings = self.text_encoder(self.texts.to(text_embeddings_device)).expand(B, -1, -1)
             if self.context_decoder:
                 if self.gamma is None: raise AttributeError("'gamma' parameter missing")
                 text_diff = self.context_decoder(text_embeddings, visual_context.to(text_embeddings_device))
                 text_embeddings = text_embeddings + self.gamma.to(text_embeddings_device) * text_diff
        else: raise TypeError(f"Unsupported text encoder type: {type(self.text_encoder)}")


        # --- Score Map Calculation ---
        B, K, C_text = text_embeddings.shape
        # Use potentially projected visual_embeddings
        visual_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)

        # Check dimensions AFTER potential projection
        # C_vis now holds the projected dimension if projection happened
        if C_vis != C_text:
             raise ValueError(f"Visual ({C_vis}) and Text ({C_text}) feature dims mismatch AFTER projection.")

        score_map = torch.einsum('bchw,bkc->bkhw', visual_norm, text_norm)

        # --- Feature Concatenation for Neck/Head ---
        features_to_process = [feat.clone() for feat in _x_orig] # Use original backbone features for neck input
        if 0 <= self.score_concat_index < len(features_to_process):
             target_feat_map = features_to_process[self.score_concat_index]
             if score_map.shape[2:] != target_feat_map.shape[2:]:
                  score_map_resized = F.interpolate(score_map, size=target_feat_map.shape[2:], mode='bilinear', align_corners=False)
             else: score_map_resized = score_map
             features_to_process[self.score_concat_index] = torch.cat([target_feat_map, score_map_resized], dim=1)
             logger.debug(f"Concatenated score map to feature index {self.score_concat_index}.")
        else: logger.warning(f"score_concat_index {self.score_concat_index} invalid. Score map not concatenated.")

        # Return original backbone features (_x_orig) for potential use by auxiliary heads
        return text_embeddings, features_to_process, score_map, _x_orig

    def forward(self, img, img_metas=None, gt_semantic_seg=None, return_loss=True, **kwargs):
        """
        Main forward pass. Determines train/inference mode.
        Args:
            img (Tensor): Input images (N, C, H, W).
            img_metas (list[dict]): List of image info dicts (Can be None for inference).
            gt_semantic_seg (Tensor): Ground truth segmentation masks (N, H, W) (for training).
            return_loss (bool): Flag indicating training mode (passed from training loop).
        """
        x = self.extract_feat(img) # Get backbone features (tuple/list)

        text_embeddings, features_for_head, score_map, _x_orig = self._process_features(x)

        # Process features through Neck if it exists
        if self.neck:
            # neck input is 'features_for_head' (list) which includes concatenated score map
            neck_outputs_dict = self.neck(features_for_head)
            # Convert neck output dict to list, ordered as expected (e.g., high-res first if neck maintains order)
            # Assuming _neck_out_keys corresponds to levels ['0', '1', '2', '3', 'pool'] or similar
            # We need to know which key corresponds to the highest resolution (smallest stride) feature
            # Usually key '0' for torchvision FPN corresponds to stride 4 (highest res)
            if self._neck_out_keys:
                 features_after_neck = [neck_outputs_dict[k] for k in self._neck_out_keys if k in neck_outputs_dict] # Get features based on keys
            else:
                 # Fallback if keys couldn't be determined
                 logger.warning("Neck output keys unknown, using values directly. Order might be incorrect.")
                 features_after_neck = list(neck_outputs_dict.values())
        else:
            # If no neck, use the features prepared by _process_features
            features_after_neck = features_for_head

        # --- START CHANGE: Select Input for FCNHead ---
        # FCNHead expects a single tensor. Use the highest resolution feature map.
        # Check if features_after_neck is a list/tuple and not empty
        if isinstance(features_after_neck, (list, tuple)) and len(features_after_neck) > 0:
             # **Assumption:** The highest resolution feature map (smallest stride, e.g., 4)
             # is the FIRST element in the features_after_neck list.
             # This is typical if the list comes directly from _process_features (backbone stages)
             # or if the neck output keys were ordered ['0', '1', '2', ...].
             # VERIFY THIS ASSUMPTION based on your backbone/neck output order.
             input_for_decode_head = features_after_neck[0]
             logger.debug(f"Selected feature map shape for decode head: {input_for_decode_head.shape}")
        elif torch.is_tensor(features_after_neck):
             # If features_after_neck is already a single tensor (e.g., no neck/FPN used)
             input_for_decode_head = features_after_neck
             logger.debug(f"Using single feature map tensor for decode head: {input_for_decode_head.shape}")
        else:
             logger.error("Could not determine input tensor for decode head from features_after_neck.")
             # Handle error: return zero loss in training, None in inference
             if return_loss and self.training:
                  return {'loss': torch.tensor(0.0, device=img.device, requires_grad=True)}
             else:
                  return None

        # Handle text_head logic - Cannot directly merge text embeddings with single tensor input easily
        if self.text_head:
            logger.warning("text_head=True ignored: Standard FCNHead expects a single feature tensor, not text embeddings.")
            # If text_head is essential, a custom head combining text and visual features is needed.
        # --- END CHANGE ---


        # --- Pass selected feature to Decode Head ---
        if not self.decode_head:
             raise RuntimeError("Decode head is not defined.")

        # Decode head's forward call returns logits (N, NumClasses, H', W')
        output_logits = self.decode_head(input_for_decode_head)

        # --- Handle Training vs Inference ---
        if return_loss and self.training:
             # --- Loss Calculation (Primary Head) ---
             # Resize logits to match GT label size BEFORE calculating loss
             gt_h, gt_w = gt_semantic_seg.shape[-2:]
             if output_logits.shape[-2:] != (gt_h, gt_w):
                  output_logits_resized = F.interpolate(
                      output_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners
                  )
             else:
                  output_logits_resized = output_logits

             # The main training loop expects a dictionary containing the loss(es)
             losses = {}
             # Calculate main decode loss using criterion from train_worker (passed implicitly)
             # losses['loss_decode'] = criterion(output_logits_resized, gt_semantic_seg) # Calculate loss here if needed

             # --- Optional: Auxiliary/Identity Heads during Training ---
             # Calculate losses for auxiliary heads if they exist and return logits
             if self.with_auxiliary_head and self.auxiliary_head:
                  # Assuming aux_head takes original backbone features (_x_orig)
                  aux_logits = self.auxiliary_head(_x_orig)
                  if aux_logits.shape[-2:] != (gt_h, gt_w):
                       aux_logits_resized = F.interpolate(aux_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners) # Check alignment
                  else:
                       aux_logits_resized = aux_logits
                  # Calculate aux loss using criterion from train_worker
                  # losses['loss_aux'] = criterion(aux_logits_resized, gt_semantic_seg) * aux_weight
                  logger.warning("Auxiliary head loss calculation not fully implemented.")
                  pass

             if self.with_identity_head and self.identity_head:
                  # Assuming identity head takes score_map / tau
                  id_logits = self.identity_head(score_map / self.tau)
                  if id_logits.shape[-2:] != (gt_h, gt_w):
                       id_logits_resized = F.interpolate(id_logits, size=(gt_h, gt_w), mode='bilinear', align_corners=self.align_corners) # Check alignment
                  else:
                       id_logits_resized = id_logits
                  # Calculate identity loss using criterion from train_worker
                  # losses['loss_identity'] = criterion(id_logits_resized, gt_semantic_seg) * id_weight
                  logger.warning("Identity head loss calculation not fully implemented.")
                  pass

             # IMPORTANT: Return the main logits for the primary loss calculation in train_worker
             # Return aux losses separately if calculated here
             return {'main_output': output_logits_resized, 'aux_losses': losses} # Return logits and aux losses

        else: # Inference mode
             # Resize output logits to match the original input image size
             output = F.interpolate(
                 input=output_logits, # Use logits from decode_head
                 size=img.shape[2:],
                 mode='bilinear',
                 align_corners=self.align_corners
             )
             return output # Return final resized logits for inference

    def inference(self, img, img_meta, rescale):
         """Simple inference, returns logits."""
         # test_cfg might control sliding window etc. - not implemented here
         seg_logit = self.forward(img, img_metas=img_meta, return_loss=False) # Call main forward in inference mode

         # Rescaling logic (if needed and not handled by forward)
         if rescale:
              ori_shape = img_meta[0]['ori_shape'][:2]
              if seg_logit.shape[2:] != ori_shape:
                  seg_logit = F.interpolate(
                      seg_logit,
                      size=ori_shape,
                      mode='bilinear',
                      align_corners=self.align_corners # Use head's setting
                  )
         return seg_logit


    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # Assuming batch size 1 for simple_test
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations by averaging logits."""
        seg_logits = [self.inference(img, meta, rescale) for img, meta in zip(imgs, img_metas)]
        avg_seg_logit = torch.stack(seg_logits).mean(dim=0)
        seg_pred = avg_seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return seg_pred

    # forward_dummy might not be needed if FLOPs calculation uses another method
    def forward_dummy(self, img):
        """ Dummy forward for FLOPs calculation or similar. """
        logger.warning("forward_dummy may not accurately reflect full model complexity.")
        # Simulate a basic forward pass without text/context for simplicity
        x = self.extract_feat(img)
        features_for_head = list(x[:-1]) # Use backbone features directly
        if self.neck:
            features_for_head = self.neck(features_for_head)
            if isinstance(features_for_head, dict): # Handle dict output from neck
                features_for_head = list(features_for_head.values())
        if self.decode_head:
            out = self.decode_head(features_for_head)
            out = F.interpolate(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            return out
        else:
            return features_for_head # Return intermediate features if no head