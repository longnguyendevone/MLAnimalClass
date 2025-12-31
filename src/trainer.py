import torch
import copy
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def train_model(model, loaders, criterion, optimizer, num_epochs=10, device='cuda'):
    scaler = torch.amp.GradScaler('cuda')
    
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_acc': [], 'val_f1': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # We will store the final epoch's predictions for the report
    final_preds = []
    final_targets = []

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(loaders['train'].dataset)
        history['train_loss'].append(epoch_loss)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Metrics
        val_loss = val_loss / len(loaders['val'].dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)

        print(f" Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save these specific predictions as the "best" for the report
            final_preds = all_preds
            final_targets = all_labels

    # Load best weights
    model.load_state_dict(best_model_wts)
    cm = confusion_matrix(final_targets, final_preds)
    
    # Return targets/preds for the classification report
    return model, history, cm, final_targets, final_preds