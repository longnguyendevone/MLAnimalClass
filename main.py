import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import gc

# Imports
from src.dataset import get_data_loaders
from src.models import get_model_architecture
from src.trainer import train_model
from src.utils import plot_comprehensive_metrics, save_hyperparameters, save_classification_report

def main():
    parser = argparse.ArgumentParser(description="Animals-10 Training Pipeline")
    parser.add_argument('--data_path', type=str, default='./data/raw-img', help='Path to Kaggle dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--models', nargs='+', default=['custom_cnn', 'mobilenet'], 
                        help='Models to train (e.g., custom_cnn, custom_transformer, deit, vgg16)')
    
    # Optimization Flags
    parser.add_argument('--no_checkpointing', action='store_false', dest='use_checkpointing',
                        help='Disable gradient checkpointing (Faster speed, higher VRAM usage)')
    parser.add_argument('--freeze', action='store_true', 
                        help='Freeze backbone layers for Transfer Learning (Lowest VRAM)')
    
    parser.set_defaults(use_checkpointing=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Load Data (Shared across models to save time)
    loaders, class_names = get_data_loaders(args.data_path, args.batch_size)
    
    # Save Class List (Global requirement)
    with open('checkpoints/classes.json', 'w') as f:
        json.dump(class_names, f)

    # 2. Training Loop
    for model_name in args.models:
        print(f"\n{'='*40}\n Training: {model_name}\n{'='*40}")

        try:
            # --- NEW: Save Hyperparameters uniquely for this model ---
            # We add the specific model name to the config to be explicit
            current_config = vars(args).copy()
            current_config['current_model_architecture'] = model_name
            
            save_hyperparameters(
                argparse.Namespace(**current_config), 
                f"results/{model_name}_config.json"
            )

            # Build Model
            model = get_model_architecture(
                model_name, 
                len(class_names), 
                pretrained=True,
                freeze_backbone=args.freeze,
                use_checkpointing=args.use_checkpointing
            )
            
            if model is None: continue
            model = model.to(device)

            # Train
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
            
            model, history, cm, y_true, y_pred = train_model(
                model, loaders, criterion, optimizer, args.epochs, device
            )

            # A. Save Weights
            torch.save(model.state_dict(), f"checkpoints/{model_name}.pth")
            
            # B. Generate Reports (Images & Text)
            # All files are now prefixed with {model_name}_
            plot_comprehensive_metrics(history, cm, class_names, f"results/{model_name}_graphs.png")
            save_classification_report(y_true, y_pred, class_names, f"results/{model_name}_report.txt")

            # C. Export to C++
            print(f"Exporting {model_name} to C++...")
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            traced = torch.jit.trace(model, dummy_input)
            traced.save(f"checkpoints/{model_name}_cpp.pt")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup VRAM
            del model
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    main()