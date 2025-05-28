import os
import torch
import matplotlib.pyplot as plt
from utils.export_results import save_predictions_to_excel  

from utils.export_results import save_predictions_to_excel  
from utils.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_curves_by_class,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_epoch_metrics_to_excel(epoch_metrics, output_path):
    import pandas as pd
    df = pd.DataFrame(epoch_metrics)
    df.to_excel(output_path, index=False)
    print(f"Epoch metrics saved to {output_path}")

def train_and_validate_model(
    train_loader, val_loader, model, criterion, optimizer,
    num_epochs=50, output_folder='model_output', class_names=None
):
    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    epoch_metrics = []

    for epoch in range(num_epochs):
      
        model.train()
        running_loss = correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

       
        model.eval()
        running_vloss = vcorrect = vtotal = 0
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                vloss = criterion(outputs, labels)
                running_vloss += vloss.item() * imgs.size(0)

                probs = torch.softmax(outputs, 1)
                preds = outputs.argmax(dim=1)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        val_loss = running_vloss / vtotal
        val_acc  = 100 * vcorrect / vtotal
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}%  "
              f"Val   loss: {val_loss:.4f}, acc: {val_acc:.2f}%")

        epoch_metrics.append({
            'Epoch': epoch+1,
            'Train Loss': train_loss,
            'Train Acc (%)': train_acc,
            'Val Loss': val_loss,
            'Val Acc (%)': val_acc,
        })

  
    save_predictions_to_excel(
        all_preds, all_labels, all_probs,
        class_names, os.path.join(output_folder, 'predictions.xlsx')
    )
    
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          normalize=True,
                          output_path=os.path.join(output_folder, 'cm.png'))
    plot_roc_curve(all_labels, all_probs, len(class_names),
                   output_path=os.path.join(output_folder, 'roc.png'))
    plot_roc_curves_by_class(all_labels, all_probs, len(class_names),
                             class_names,
                             output_path=os.path.join(output_folder, 'roc_by_class.png'))
    
    save_epoch_metrics_to_excel(epoch_metrics,
                                os.path.join(output_folder, 'epoch_metrics.xlsx'))

    
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accs, label='Train')
    plt.plot(range(1, num_epochs+1), val_accs,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend()
    plt.savefig(os.path.join(output_folder, 'acc_curve.png')); plt.close()

  
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train')
    plt.plot(range(1, num_epochs+1), val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(output_folder, 'loss_curve.png')); plt.close()

  
    torch.save(model.state_dict(), os.path.join(output_folder, 'model.pth'))
    print(f"All outputs saved to `{output_folder}`.")
