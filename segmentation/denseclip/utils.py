import os
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import gzip
import html
import ftfy
import regex as re
from functools import lru_cache
import platform
import subprocess

from pathlib import Path

# Add this near the tokenizer code
@lru_cache()
def default_bpe():
    return str(Path(__file__).parent / "bpe_simple_vocab_16e6.txt.gz")

# Modify tokenizer initialization
#_tokenizer = SimpleTokenizer(bpe_path=default_bpe())


# ==================== TRAINING UTILITIES ==================== #
def setup_logger(log_dir: str, rank: int = 0) -> logging.Logger:
    """Initialize rank-aware logger for distributed training"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('DenseCLIP')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARN)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler (all ranks)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'training_rank{rank}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def set_random_seed(seed: int, deterministic: bool = False):
    """Set reproducible training conditions"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    config: Dict[str, Any],
    best_metric: Optional[float] = None,
    is_best: bool = False
):
    """Save training checkpoint with metadata"""
    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'best_metric': best_metric
    }
    
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'model_best.pth')
        torch.save(state, best_path)

def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Load checkpoint with safety checks"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint

def init_distributed(rank: int, world_size: int):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def compute_segmentation_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> Dict[str, float]:
    """Compute pixel accuracy and mIoU for segmentation"""
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        valid_mask = (targets != ignore_index)
        
        # Pixel Accuracy
        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        acc = correct / (total + 1e-5)
        
        # Confusion Matrix
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=outputs.device)
        for t, p in zip(targets[valid_mask].view(-1), preds[valid_mask].view(-1)):
            cm[t.long(), p.long()] += 1
        
        # IoU Calculation
        intersection = torch.diag(cm)
        union = cm.sum(0) + cm.sum(1) - intersection
        iou = (intersection / (union + 1e-8)).cpu().numpy()
        
        return {
            'accuracy': acc,
            'mean_iou': np.nanmean(iou),
            'class_iou': iou
        }

# ==================== ENV COLLECTOR ==================== #
def collect_env_info():
    env_str = "Collecting environment information..."
    version_str = ""

    try:
        version_str = f"PyTorch: {torch.__version__} \n"
    except Exception:
        version_str += "PyTorch: Error \n"

    try:
        cuda_available_str = f"CUDA available: {torch.cuda.is_available()}\n"
    except Exception:
        cuda_available_str = f"CUDA available: Error \n"

    try:
        cuda_version_str = f"CUDA_VERSION: {torch.version.cuda} \n"
    except Exception:
        cuda_version_str = f"CUDA_VERSION: Error \n"

   # try:
   #     cudnn_version_str = f"CUDNN_VERSION: {torch.backends.cudnn.version()} \n"
   # except Exception:
   #     cudnn_version_str = f"CUDNN_VERSION: Error \n"

    try:
        os_name = platform.system()
        os_version = platform.release()
        os_str = f"OS: {os_name} {os_version} \n"
    except Exception:
        os_str = f"OS: Error \n"

    try:
        gcc_version = subprocess.check_output(['gcc', '-version']).decode('utf-8').split('\n')[0]
        gcc_str = f"GCC: {gcc_version} \n"
    except Exception:
        gcc_str = f"GCC: Error \n"
    
    env_info = {"System Information" : os_str, "Torch Information": version_str, "CUDA Information": cuda_available_str, "CUDA version": cuda_version_str, "GCC Info": gcc_str}

    for k, v in env_info.items():
        env_str += f"{k} : {v}"

    return env_str

# ==================== CLIP TEXT PROCESSING ==================== #
@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class SimpleTokenizer:
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# Place tokenizer instantiation here, after the class definition
_tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """Tokenize text for CLIP model input (unchanged from original)"""
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result