import argparse
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import _Linear
import yaml
from denseclip import build_model  # Assume we have this builder function
from datasets.ade20k import ADE20KSegmentation  # Custom dataset class

def calc_flops(model, input_shape=(3, 224, 224)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    with torch.no_grad():
        # Calculate FLOPs
        flops = FlopCountAnalysis(model, dummy_input)
        
        # Print FLOPs breakdown
        print("\nFLOPs Breakdown:")
        print('-' * 50)
        for module_name in ["backbone", "text_encoder", "context_decoder", "neck", "decode_head"]:
            try:
                module_flops = flops.by_module()[f"module.{module_name}"] if hasattr(model, "module") else flops.by_module()[module_name]
                print(f"{module_name:15s}: {module_flops/1e9:.2f} GFLOPs")
            except KeyError:
                continue
        
        total_flops = flops.total()
        print('-' * 50)
        print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")
        
        return total_flops

def count_parameters(model):
    """Count number of trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown by components
    print("\nParameter Count Breakdown:")
    print('-' * 50)
    for name, module in model.named_modules():
        if isinstance(module, (_ConvNd, _Linear, _BatchNorm)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:40s}: {params/1e6:.2f} M")
    
    print('-' * 50)
    print(f"Total Parameters: {total_params/1e6:.2f} M")
    return total_params

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FLOPs and parameters for DenseCLIP')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--shape', type=int, nargs='+', 
                       default=[512, 512], help='Input image size (H W)')
    parser.add_argument('--dataset', default='ade20k', 
                       help='Dataset name for class names')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Determine input shape
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    else:
        input_shape = (3, args.shape[0], args.shape[1])
    
    # Build model
    model = build_model(cfg['model'])
    
    # Get class names if needed
    if 'DenseCLIP' in cfg['model']['type']:
        if args.dataset == 'ade20k':
            dataset = ADE20KSegmentation(root=cfg['data']['data_root'], split='train')
            cfg['model']['class_names'] = dataset.CLASSES
    
    model = model.cuda().eval()
    
    # Calculate FLOPs
    print(f"\nCalculating FLOPs for input shape: {input_shape}")
    flops = calc_flops(model, input_shape)
    
    # Calculate parameters
    params = count_parameters(model)
    
    # Calculate FLOPs/parameter ratio
    print(f"\nFLOPs/Parameter Ratio: {flops/params:.4f}")

if __name__ == '__main__':
    main()