import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define correct model order and colors
MODEL_ORDER = [
    'openai/whisper-tiny',
    'openai/whisper-small',
    'openai/whisper-base',
    'openai/whisper-medium',
    'openai/whisper-large-v2'
]
COLORS = {'Default': 'black', 'Winograd': '#8B0000'}  # Black and dark red

# Load and preprocess data
win_df = pd.read_csv('winograd_benchmark_results.csv')
win_df['config'] = np.where(win_df['winograd_mode'], 'Winograd', 'Default')

# Enforce model order
win_df['model'] = pd.Categorical(win_df['model'], categories=MODEL_ORDER, ordered=True)
win_df = win_df.sort_values('model')

# Aggregate metrics
metrics = win_df.groupby(['model', 'config'], observed=True).agg({
    'time_sec': 'mean',
    'actual_gflops': 'mean',
    'vram_mb': 'mean'
}).reset_index()

# Calculate differences for annotation
time_diff = metrics.pivot(index='model', columns='config', values='time_sec').diff(axis=1)['Winograd']
gflops_diff = metrics.pivot(index='model', columns='config', values='actual_gflops').diff(axis=1)['Winograd']
vram_diff = metrics.pivot(index='model', columns='config', values='vram_mb').diff(axis=1)['Winograd']

# Plot configuration
plt.figure(figsize=(15, 10))
x = np.arange(len(MODEL_ORDER))
width = 0.35

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Time comparison
for i, config in enumerate(['Default', 'Winograd']):
    config_data = metrics[metrics['config'] == config]
    axs[0].bar(x + i*width, config_data['time_sec'], width, 
               color=COLORS[config], label=config)
    
    # Add difference annotations
    if config == 'Winograd':
        for j, val in enumerate(time_diff):
            diff_text = f"{val:.2f}s" if not np.isnan(val) else ""
            axs[0].text(x[j] + width/2, config_data['time_sec'].iloc[j] + 0.05, 
                        diff_text, ha='center', va='bottom', fontsize=9)

axs[0].set_title('Inference Time Comparison (Winograd)', fontweight='bold')
axs[0].set_ylabel('Seconds')
axs[0].set_xticks(x + width/2)
axs[0].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[0].legend()
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# GFLOPS comparison
for i, config in enumerate(['Default', 'Winograd']):
    config_data = metrics[metrics['config'] == config]
    axs[1].bar(x + i*width, config_data['actual_gflops'], width, 
               color=COLORS[config], label=config)
    
    if config == 'Winograd':
        for j, val in enumerate(gflops_diff):
            diff_text = f"{val:.1f}" if not np.isnan(val) else ""
            axs[1].text(x[j] + width/2, config_data['actual_gflops'].iloc[j] + 20, 
                        diff_text, ha='center', va='bottom', fontsize=9)

axs[1].set_title('Actual GFLOPS Comparison (Winograd)', fontweight='bold')
axs[1].set_ylabel('GFLOPS')
axs[1].set_xticks(x + width/2)
axs[1].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# VRAM comparison
for i, config in enumerate(['Default', 'Winograd']):
    config_data = metrics[metrics['config'] == config]
    axs[2].bar(x + i*width, config_data['vram_mb'], width, 
               color=COLORS[config], label=config)
    
    if config == 'Winograd':
        for j, val in enumerate(vram_diff):
            diff_text = f"{val:.1f}MB" if not np.isnan(val) else ""
            axs[2].text(x[j] + width/2, config_data['vram_mb'].iloc[j] + 50, 
                        diff_text, ha='center', va='bottom', fontsize=9)

axs[2].set_title('VRAM Usage Comparison (Winograd)', fontweight='bold')
axs[2].set_ylabel('MB')
axs[2].set_xticks(x + width/2)
axs[2].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('winograd_comparison.png', bbox_inches='tight', dpi=300)
plt.close()