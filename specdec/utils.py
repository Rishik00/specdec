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
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.time_elapsed = self.end_time - self.start_time

def plot_timings(baseline, speculative):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(baseline))

    plt.figure(figsize=(8, 4))
    plt.plot(x, baseline, label="Baseline", linewidth=2)
    plt.plot(x, speculative, label="Speculative", linewidth=2)
    plt.xlabel("Run Index")
    plt.ylabel("Time (ms)")
    plt.title("Baseline vs Speculative Decoding Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
