import time
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROMPTS = [
    "Explain transformers in one sentence."
    "Summarize the concept of entropy."
    "What is speculative decoding?"
    "Define gradient clipping."
    "Explain diffusion models simply."

    "Write a short explanation of how attention works in transformer models, but avoid equations."
    "Explain the difference between self-attention and cross-attention in simple terms."
    "Describe how GPU memory is used during inference in large language models."
    "Explain why increasing sequence length can slow down attention computation."
    "Give a short overview of knowledge distillation and why it's useful."

    "Talk to me like I'm your collaborator and explain why decoding is sequential and can't be parallelized entirely."
    "Give me the 'tell me straight' explanation of why speculative decoding helps, without softening anything."
    "Explain the trade-off between draft model size and acceptance rate in speculative decoding, and be blunt about when it's not worth it."
]


class Timer:

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self):
        self.end = time.per_counter()
        self.time_elapsed = self.end_time - self.start_time

def plot_times(baseline_timings, speculative_timings):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Line plot comparing timings
    x = np.arange(len(baseline_timings))
    ax1.plot(x, baseline_timings, marker='o', linewidth=2, markersize=8, 
             label='Baseline', color='#e74c3c')
    ax1.plot(x, speculative_timings, marker='s', linewidth=2, markersize=8, 
             label='Speculative', color='#2ecc71')
    ax1.set_xlabel('Run Index', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Timing Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot for distribution
    data = [baseline_timings, speculative_timings]
    bp = ax2.boxplot(data, labels=['Baseline', 'Speculative'], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Timing Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add speedup text
    avg_baseline = np.mean(baseline_timings)
    avg_speculative = np.mean(speculative_timings)
    speedup = avg_baseline / avg_speculative
    fig.suptitle(f'Average Speedup: {speedup:.2f}×', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('plot.png')
    plt.tight_layout()
    plt.show()