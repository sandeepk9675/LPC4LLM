import matplotlib.pyplot as plt
import json
import os

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'Oracle'}

def plot_dataset(file_path, y_label='mean_ttft', x_label='size'):
    with open(file_path, 'r') as file:
        data = json.load(file)

    dataset_name = data[0]['dataset_name'] if len(data) > 0 else 'Unknown'

    def get_algo_data(algo_prefix):
        xs, ys = [], []
        # Reversed so we process the latest appended results per config first
        for config in reversed(data):
            algorithm = config.get('algorithm', '')
            
            if algo_prefix == 'lru' and 'lru' not in algorithm:
                continue
            if algo_prefix == 'ml':
                if 'ml' not in algorithm or 'true' in algorithm:
                    continue
            
            x = config.get(x_label, 0)
            if 'ml' in algorithm:
                x += 250
            x = x * 16 / 1000

            if x not in xs:
                y = float(config.get(y_label, 0))
                xs.append(x)
                ys.append(y)
                
        if not xs:
            return [], []
        sorted_pairs = sorted(zip(xs, ys))
        return zip(*sorted_pairs)

    xs_lru, ys_lru = get_algo_data('lru')
    xs_lpc, ys_lpc = get_algo_data('ml')

    # Plotting
    plt.figure(figsize=(3.5, 2.7))
    if xs_lru:
        plt.plot(xs_lru, ys_lru, marker='o', label='LRU')
    if xs_lpc:
        plt.plot(xs_lpc, ys_lpc, marker='s', label='LPC')

    plt.xlabel('Cache Size (×10³ tokens)')
    if y_label == 'mean_ttft':
        plt.ylabel('Mean TTFT (ms)')
    elif y_label == 'p99_ttft':
        plt.ylabel('P99 TTFT (ms)')
    else:    
        plt.ylabel('Hit Ratio')
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save figure to file
    os.makedirs('fig2', exist_ok=True)
    out_path = f'fig2/{y_label}_{dataset_name}_LPCvsLRU.png'
    print(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    # plt.show()

dir_meta = "/home/sandeep/snap/Academics/SysML/proj1/LPC/vllm_cache_bench/results/Llama-3.1-8B-Instruct"

file_paths_meta = [
    f'{dir_meta}/exp_lmsys.json',
    f'{dir_meta}/exp_chatbot.json',
    f'{dir_meta}/exp_sharegpt.json',
]

for f_meta in file_paths_meta:
    try:
        plot_dataset(f_meta, 'mean_ttft', 'size')
    except Exception as e:
        print(f"Skipping {f_meta} due to error: {e}")
