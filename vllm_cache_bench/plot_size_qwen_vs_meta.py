import matplotlib.pyplot as plt
import json

name = {'ml': 'LPC', 'lru': 'LRU', 'ml-true': 'Oracle'}

def plot_dataset(file_path_qwen, file_path_meta, y_label, x_label):
    with open(file_path_qwen, 'r') as file:
        data_qwen = json.load(file)

    with open(file_path_meta, 'r') as file:
        data_meta = json.load(file)

    dataset_name = data_qwen[0]['dataset_name']

    def get_lru_data(data):
        xs, ys = [], []
        for config in reversed(data):
            algorithm = config.get('algorithm', '')
            if 'lru' not in algorithm:
                continue
            x = config[x_label]
            x = x * 16 / 1000
            if x not in xs:
                if y_label == 'hit_ratios':
                    y = float(config[y_label][-1])
                else:
                    y = float(config[y_label])
                xs.append(x)
                ys.append(y)
        if not xs:
            return [], []
        sorted_pairs = sorted(zip(xs, ys))
        return zip(*sorted_pairs)

    xs_qwen, ys_qwen = get_lru_data(data_qwen)
    xs_meta, ys_meta = get_lru_data(data_meta)

    # Plotting
    plt.figure(figsize=(3.5, 2.7))
    if xs_qwen:
        plt.plot(xs_qwen, ys_qwen, marker='o', label='Qwen (LRU)')
    if xs_meta:
        plt.plot(xs_meta, ys_meta, marker='s', label='Llama (LRU)')

    plt.ylim(bottom=0.04)
    plt.ylim(top=0.7)
    plt.xlabel('Cache Size (×10³ tokens)')
    if y_label == 'p99_ttft':
        plt.ylabel('P99 TTFT (ms)')
    else:    
        plt.ylabel('Hit Ratio')
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save figure to file
    import os
    os.makedirs('fig2', exist_ok=True)
    print(f'fig2/{y_label}_{dataset_name}.png')
    plt.savefig(f'fig2/{y_label}_{dataset_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    # plt.show()

# dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/'
dir_meta = "/home/sandeep/snap/Academics/SysML/proj1/LPC/vllm_cache_bench/results/Llama-3.1-8B-Instruct"
dir_qwen = 'results/98a63908b41686889a6ade39c37616e54d49974d'
#dir = 'results/98a63908b41686889a6ade39c37616e54d49974d/result_Nconv=300_1'
file_paths_meta = [
    f'{dir_meta}/exp_lmsys.json',
    f'{dir_meta}/exp_chatbot.json',
    f'{dir_meta}/exp_sharegpt.json',
]

file_paths_qwen = [
    f'{dir_qwen}/exp_lmsys_0514.json',
    f'{dir_qwen}/exp_chatbot_0514.json',
    f'{dir_qwen}/exp_sharegpt_0514.json',
]   
#file_paths = [f'{dir}/exp_chatbot200.json',
#f'{dir}/exp_sharegpt200.json',
#f'{dir}/exp_lmsys200.json',
#]

for f_meta, f_qwen in zip(file_paths_meta, file_paths_qwen):
    try:
        plot_dataset(f_qwen, f_meta, 'hit_ratios', 'size')
    except Exception as e:
        print(f"Skipping {f_qwen} and {f_meta} due to error: {e}")

# 200 does not have the conversaton end time adjustment by total output length
