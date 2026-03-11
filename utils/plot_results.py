import matplotlib.pyplot as plt
import numpy as np

def plot_results(all_histories, accuracies):
    n_subjects = len(all_histories)
    
    # Plot 1: Train vs Test accuracy curves per subject
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Train vs Test Accuracy per Subject', fontsize=16)
    
    for i, (ax, data) in enumerate(zip(axes.flatten(), all_histories)):
        subject = data['subject']
        history = data['history']
        
        ax.plot(history['epochs'], history['train_acc'], label='Train', color='blue')
        ax.plot(history['epochs'], history['test_acc'], label='Test', color='orange')
        ax.axhline(y=0.25, color='gray', linestyle='--', label='Chance')
        ax.set_title(f'Subject {subject}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('utils/accuracy_curves.png', dpi=150)
    plt.show()

    # Plot 2: Loss curves per subject
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Loss per Subject', fontsize=16)
    
    for i, (ax, data) in enumerate(zip(axes.flatten(), all_histories)):
        subject = data['subject']
        history = data['history']
        
        ax.plot(history['epochs'], history['loss'], color='red')
        ax.set_title(f'Subject {subject}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('utils/loss_curves.png', dpi=150)
    plt.show()

    # Plot 3: Bar chart of per-subject accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    subjects = [d['subject'] for d in all_histories]
    colors = ['green' if a >= 0.70 else 'orange' if a >= 0.55 else 'red' for a in accuracies]
    
    bars = ax.bar([f'S{s}' for s in subjects], accuracies, color=colors)
    ax.axhline(y=np.mean(accuracies), color='blue', linestyle='--', 
               label=f'Mean: {np.mean(accuracies):.3f}')
    ax.axhline(y=0.25, color='gray', linestyle='--', label='Chance (25%)')
    ax.axhline(y=0.72, color='purple', linestyle='--', label='Paper (~72%)')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Per-Subject Classification Accuracy')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('utils/subject_accuracy.png', dpi=150)
    plt.show()