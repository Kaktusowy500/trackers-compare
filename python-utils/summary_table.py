import os
import yaml
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np


def process_summary_file(summary_file):
    with open(summary_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def calculate_averages(results):
    return {tracker: {key: round(np.mean(values), 5) for key, values in metrics.items()} for tracker, metrics in results.items()}


def save_table_as_image(df, output_file_img, exclude_columns):
    fig, ax = plt.subplots(figsize=(10, 4)) 
    ax.axis('tight')
    ax.axis('off')

    if exclude_columns:
        df = df.drop(columns=exclude_columns)

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Apply cell colors for max/min values
    for i, row in enumerate(df.itertuples()):
        for j, val in enumerate(row[1:]):
            cell = table[(i + 1, j)]
            if df.columns[j] in ['average_overlap', 'valid_frame_percent']:
                if val == df[df.columns[j]].max():
                    cell.set_facecolor('#a1d99b')
            else:
                if val == df[df.columns[j]].min():
                    cell.set_facecolor('#a1d99b')

    plt.savefig(output_file_img, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Results table saved as an image to {output_file_img}')


def main(base_dir):
    config_file = os.path.join(base_dir, 'config.yaml')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    reinit_strategy = config.get('reinit_strategy', '')
    exclude_columns = ['reinit_count'] if reinit_strategy == 'one_init' else []

    overall_results = {
        'CSRT': {'average_overlap': [], 'average_error': [], 'average_processing_time': [], 'valid_frame_percent': [], 'reinit_count': []},
        'VIT': {'average_overlap': [], 'average_error': [], 'average_processing_time': [], 'valid_frame_percent': [], 'reinit_count': []},
        'ModVIT': {'average_overlap': [], 'average_error': [], 'average_processing_time': [], 'valid_frame_percent': [], 'reinit_count': []},
        'DaSiam': {'average_overlap': [], 'average_error': [], 'average_processing_time': [], 'valid_frame_percent': [], 'reinit_count': []},
    }

    # Create plots directory if it does not exist
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Process each subfolder
    for root, dirs, files in os.walk(base_dir):
        if 'summary.yaml' in files:
            summary_file = os.path.join(root, 'summary.yaml')
            data = process_summary_file(summary_file)

            subfolder_results = {tracker: {key: data[tracker][key] for key in overall_results[tracker].keys()} for tracker in overall_results.keys()}
            subfolder_averages = calculate_averages(subfolder_results)
            subfolder_df = pd.DataFrame(subfolder_averages).T
            print(f"Results for {root}:\n{subfolder_df}\n")
            subfolder_name = os.path.basename(root)
            subfolder_output_file_img = os.path.join(plots_dir, f'{subfolder_name}_average_results.png')
            save_table_as_image(subfolder_df, subfolder_output_file_img, exclude_columns)

            # Append current subfolder results to the overall results
            for tracker in overall_results.keys():
                for key in overall_results[tracker].keys():
                    overall_results[tracker][key].append(data[tracker][key])

    overall_averages = calculate_averages(overall_results)
    overall_df = pd.DataFrame(overall_averages).T
    print(f"Overall Results:\n{overall_df}\n")
    overall_output_file_img = os.path.join(plots_dir, 'overall_average_tracker_results.png')
    save_table_as_image(overall_df, overall_output_file_img, exclude_columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML files and compare tracker results.")
    parser.add_argument('base_dir', type=str, help="Base directory to scan for summary.yaml files.")
    args = parser.parse_args()
    main(args.base_dir)
