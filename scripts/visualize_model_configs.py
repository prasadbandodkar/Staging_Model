import matplotlib.pyplot as plt
import numpy as np

def visualize_model_configs():
    # Configuration Data
    model_configs = {
        'nano':   {'channels': 8,  'blocks': (1, 1, 1, 1), 'params': '< 150k'},
        'tiny':   {'channels': 32, 'blocks': (1, 1, 1, 1), 'params': '~ 700k'},
        'small':  {'channels': 64, 'blocks': (2, 2, 2, 2), 'params': '~ 11M'},
        'medium': {'channels': 64, 'blocks': (3, 4, 6, 3), 'params': '~ 25M'},
        'large':  {'channels': 64, 'blocks': (3, 4, 23, 3), 'params': 'Distinguished by Depth'}
    }

    models = list(model_configs.keys())
    channels = [model_configs[m]['channels'] for m in models]
    blocks_l1 = [model_configs[m]['blocks'][0] for m in models]
    blocks_l2 = [model_configs[m]['blocks'][1] for m in models]
    blocks_l3 = [model_configs[m]['blocks'][2] for m in models]
    blocks_l4 = [model_configs[m]['blocks'][3] for m in models]

    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.style.use('bmh') # Clean aesthetic

    # Plot 1: Base Width (Channels)
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0']
    bars = ax1.bar(models, channels, color=colors, edgecolor='black', alpha=0.9)
    ax1.set_title('Model Width: Base Channels', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Base Filters', fontsize=12)
    ax1.set_ylim(0, 80)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add text box for param counts explanation
    param_text = "Approx. Parameters:\n" + "\n".join([f"{m.title()}: {model_configs[m]['params']}" for m in models])
    ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Model Depth (Stacked Blocks)
    # Stacking
    b1 = np.array(blocks_l1)
    b2 = np.array(blocks_l2)
    b3 = np.array(blocks_l3)
    b4 = np.array(blocks_l4)

    p1 = ax2.bar(models, b1, label='Layer 1', color='#1f77b4', edgecolor='white')
    p2 = ax2.bar(models, b2, bottom=b1, label='Layer 2', color='#ff7f0e', edgecolor='white')
    p3 = ax2.bar(models, b3, bottom=b1+b2, label='Layer 3', color='#2ca02c', edgecolor='white')
    p4 = ax2.bar(models, b4, bottom=b1+b2+b3, label='Layer 4', color='#d62728', edgecolor='white')

    ax2.set_title('Model Depth: Blocks per Layer', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Number of Residual Blocks', fontsize=12)
    ax2.legend(title="ResNet Layers", loc='upper left')

    # Add total blocks labels
    totals = b1 + b2 + b3 + b4
    for i, t in enumerate(totals):
        ax2.text(i, t + 0.5, f'Total: {t}', ha='center', va='bottom', fontweight='bold')

    # Overall Formatting
    plt.suptitle('Model Architecture Comparison: Width vs. Depth', fontsize=18, y=0.98)
    plt.tight_layout()
    
    # Save
    output_path = 'model_architecture_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    visualize_model_configs()
