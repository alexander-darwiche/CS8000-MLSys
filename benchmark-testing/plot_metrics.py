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
COLORS = {'Full Precision': 'black', 'Quantized': '#8B0000'}  # Black and dark red

# Load and preprocess data
quant_df = pd.read_csv('quant_benchmark_results2.csv')
quant_df['config'] = np.where(quant_df['quant_mode'] == 'fp16', 'Quantized', 'Full Precision')

# Enforce model order
quant_df['model'] = pd.Categorical(quant_df['model'], categories=MODEL_ORDER, ordered=True)
quant_df = quant_df.sort_values('model')

# Aggregate metrics
metrics = quant_df.groupby(['model', 'config'], observed=True).agg({
    'time_sec': 'mean',
    'wer': 'mean',
    'vram_mb': 'mean'
}).reset_index()

# Calculate differences for annotation
time_diff = metrics.pivot(index='model', columns='config', values='time_sec').diff(axis=1)['Quantized']
wer_diff = metrics.pivot(index='model', columns='config', values='wer').diff(axis=1)['Quantized']
vram_diff = metrics.pivot(index='model', columns='config', values='vram_mb').diff(axis=1)['Quantized']

# Plot configuration
plt.figure(figsize=(15, 10))
x = np.arange(len(MODEL_ORDER))
width = 0.35

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(14, 18))

# Time comparison
for i, config in enumerate(['Full Precision', 'Quantized']):
    config_data = metrics[metrics['config'] == config]
    bars = axs[0].bar(x + i*width, config_data['time_sec'], width, 
                    color=COLORS[config], label=config)
    
    # Add difference annotations
    if config == 'Quantized':
        for j, val in enumerate(time_diff):
            diff_text = f"{val:.2f}s" if not np.isnan(val) else ""
            bar_height = config_data['time_sec'].iloc[j]
            axs[0].text(x[j] + width/2,  # Center horizontally
                        bar_height + (0.1 * bar_height),  # Position 10% above bar
                        diff_text, 
                        ha='center', 
                        va='bottom',
                        fontsize=9,
                        color=COLORS[config],
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

axs[0].set_title('Inference Time Comparison (Quantization)', fontweight='bold')
axs[0].set_ylabel('Seconds')
axs[0].set_xticks(x + width/2)
axs[0].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[0].legend()
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# WER comparison
for i, config in enumerate(['Full Precision', 'Quantized']):
    config_data = metrics[metrics['config'] == config]
    bars = axs[1].bar(x + i*width, config_data['wer'], width, 
                    color=COLORS[config], label=config)
    
    if config == 'Quantized':
        for j, val in enumerate(wer_diff):
            diff_text = f"{val:.4f}" if not np.isnan(val) else ""
            bar_height = config_data['wer'].iloc[j]
            axs[1].text(x[j] + width/2,
                        bar_height + (0.15 * bar_height),  # Position 15% above bar
                        diff_text,
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color=COLORS[config],
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

axs[1].set_title('Word Error Rate Comparison (Quantization)', fontweight='bold')
axs[1].set_ylabel('WER')
axs[1].set_xticks(x + width/2)
axs[1].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# VRAM comparison
for i, config in enumerate(['Full Precision', 'Quantized']):
    config_data = metrics[metrics['config'] == config]
    bars = axs[2].bar(x + i*width, config_data['vram_mb'], width, 
                    color=COLORS[config], label=config)
    
    if config == 'Quantized':
        for j, val in enumerate(vram_diff):
            diff_text = f"{val:.1f}MB" if not np.isnan(val) else ""
            bar_height = config_data['vram_mb'].iloc[j]
            axs[2].text(x[j] + width/2,
                        bar_height + (0.05 * bar_height),  # Position 5% above bar
                        diff_text,
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color=COLORS[config],
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

axs[2].set_title('VRAM Usage Comparison (Quantization)', fontweight='bold')
axs[2].set_ylabel('MB')
axs[2].set_xticks(x + width/2)
axs[2].set_xticklabels([m.split('/')[-1] for m in MODEL_ORDER], rotation=45)
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('quantization_comparison.png', bbox_inches='tight', dpi=300)
plt.close()