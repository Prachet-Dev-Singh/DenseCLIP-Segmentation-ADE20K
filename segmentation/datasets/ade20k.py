import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ADE20KSegmentation(Dataset):
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror',
        'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
    )
    PALETTE = np.random.randint(0, 255, size=(len(CLASSES), 3))

    def __init__(self, root, split='training', crop_size=512, scale=(0.5, 2.0), ignore_label=255):
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        
        if split == 'training':
            self.image_dir = os.path.join(root, 'ADEChallengeData2016', 'images', 'training')
            self.label_dir = os.path.join(root, 'ADEChallengeData2016', 'annotations', 'training')
        else:
            self.image_dir = os.path.join(root, 'ADEChallengeData2016', 'images', 'validation')
            self.label_dir = os.path.join(root, 'ADEChallengeData2016', 'annotations', 'validation')
        
        self.filenames = [f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)
    
    _printed_label_info = False
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx] + '.jpg')
        label_name = os.path.join(self.label_dir, self.filenames[idx] + '.png')

        try: # Add try-except for file loading
            image = Image.open(img_name).convert('RGB')
            label = Image.open(label_name)
        except FileNotFoundError as e:
            print(f"Error loading file for index {idx}: {e}")
            # Return None or raise error, depending on how you want to handle missing files
            # For now, let's raise it to stop execution clearly
            raise e
        except Exception as e:
            print(f"Error processing image/label for index {idx} ({img_name}/{label_name}): {e}")
            raise e


        # Random scaling
        scale = np.random.uniform(*self.scale)
        orig_w, orig_h = image.size
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        # Use ANTIALIAS for downsampling images if using Pillow >= 8.0
        try:
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
        except AttributeError: # Fallback for older Pillow
            image = image.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST) # Always NEAREST for labels

        w, h = image.size

        # Ensure image is large enough for cropping
        if isinstance(self.crop_size, int):
            crop_h = crop_w = self.crop_size
        else:
            crop_h, crop_w = self.crop_size

        # --- START CHANGE: Simpler Resize/Padding Logic ---
        # If image is smaller than crop in any dimension, resize to *at least* crop size
        # while maintaining aspect ratio (approximately). This avoids tiny crops.
        # A common approach is to resize the *smaller* edge to crop size.
        if w < crop_w or h < crop_h:
            # Calculate aspect ratio
            aspect_ratio = w / h
            target_w, target_h = w, h
            # Resize smaller edge to crop_size
            if w < crop_w and h < crop_h: # Both smaller
                 if w/crop_w < h/crop_h: # Width is proportionally smaller
                      target_w = crop_w
                      target_h = int(target_w / aspect_ratio)
                 else: # Height is proportionally smaller or equal
                      target_h = crop_h
                      target_w = int(target_h * aspect_ratio)
            elif w < crop_w: # Only width smaller
                 target_w = crop_w
                 target_h = int(target_w / aspect_ratio)
            else: # Only height smaller
                 target_h = crop_h
                 target_w = int(target_h * aspect_ratio)

            # Ensure the *other* dimension is also at least crop size after resize
            target_w = max(target_w, crop_w)
            target_h = max(target_h, crop_h)

             # logger.debug(f"Resizing image {idx} before crop from ({w},{h}) to ({target_w},{target_h})")
            try:
                image = image.resize((target_w, target_h), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
            except AttributeError:
                image = image.resize((target_w, target_h), Image.BILINEAR)
            label = label.resize((target_w, target_h), Image.NEAREST)
            w, h = image.size # Update dimensions
        # --- END CHANGE ---


        # Random Cropping
        x_max = max(0, w - crop_w)
        y_max = max(0, h - crop_h)
        x = np.random.randint(0, x_max + 1)
        y = np.random.randint(0, y_max + 1)
        image = image.crop((x, y, x + crop_w, y + crop_h))
        label = label.crop((x, y, x + crop_h, y + crop_h))

        # --- Convert Label to Numpy and CHECK/FIX range ---
        label_np = np.array(label).astype(np.int64)

        # --- Add check for label range (run this once) ---
        """ if not ADE20KSegmentation._printed_label_info:
            min_val, max_val = np.min(label_np), np.max(label_np)
            unique_vals = np.unique(label_np)
            print(f"--- Label Info (Index {idx}, First Check Only) ---")
            print(f"Original Min Label: {min_val}")
            print(f"Original Max Label: {max_val}")
            print(f"Original Unique Labels (Sample): {unique_vals[:20]}...") # Print first 20 unique
            print(f"Config ignore_label: {self.ignore_label}")
            print("-------------------------------------------------")
            ADE20KSegmentation._printed_label_info = True # Set flag so it doesn't print every time """
        # --- End check ---


        # --- START CHANGE: Correct Label Mapping for ADE20K (0:ignore, 1-150:classes mapped to 0-149) ---
        if self.ignore_label is None:
             # If ignore_label is not set, treat 0 as a class (problematic for ADE20K)
             warnings.warn("ignore_label is None. ADE20K background (0) will be treated as a class.")
             # If 0 should be class 0, labels 1-150 become 1-150 (requires num_classes=151)
             # This usually isn't standard ADE20K setup.
             pass # No mapping needed if 0 is a valid class
        else:
             # Standard ADE20K: Map 0 to ignore_index, map 1-150 to 0-149
             valid_mask = (label_np > 0) # Select original labels 1-150
             label_np[valid_mask] = label_np[valid_mask] - 1 # Map 1-150 --> 0-149
             label_np[~valid_mask] = self.ignore_label # Map 0 --> ignore_label (e.g., 255)
        # --- END CHANGE ---

        # --- Final Check (Optional Debugging) ---
        # if np.any(label_np >= 150) and np.any(label_np != self.ignore_label) :
        #      print(f"WARNING: Label contains values >= 150 after mapping (Index {idx})! Max: {label_np.max()}")
        # if np.any(label_np < 0):
        #      print(f"WARNING: Label contains values < 0 after mapping (Index {idx})! Min: {label_np.min()}")
        # --- End Final Check ---


        # Convert image to numpy, normalize, transpose
        image_np = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]) # Use appropriate mean/std
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        image_np = image_np.transpose(2, 0, 1) # HWC to CHW


        return torch.from_numpy(image_np), torch.from_numpy(label_np) # Return modified label_np
