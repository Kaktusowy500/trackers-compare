import os
import yaml
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def process_summary_file(summary_file):
    with open(summary_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def round_results(results):
    r_results = {}
    for tracker, metrics in results.items():
        r_results[tracker] = {}
        for key, values in metrics.items():
            digits = 5 if "time" in key else 4
            r_results[tracker][key] = round(values, digits)
    return r_results


def calculate_overall_averages_and_std(results):
    final_res = {}
    for tracker, metrics in results.items():
        final_res[tracker] = {}
        for key, values in metrics.items():
            digits = 5 if "time" in key else 4
            if not "std" in key:
                final_res[tracker][key] = round(np.mean(values), digits)
                final_res[tracker][key + "_std"] = round(np.std(values), digits)
    return final_res


def save_table_as_image(df, output_file_img, exclude_columns, overall=False):
    # Adjust the column names for std columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')
    if overall:
        df = df.rename(columns={'SR': 'avg_SR', 'RC': 'avg_RC'})

    df.columns = [col if "std" not in col else "std" for col in df.columns]
    fig, ax = plt.subplots(figsize=(10, 2))  # Adjusted the size for better readability
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Adjust the column widths for std columns
    std_cols = [j for j, col in enumerate(df.columns) if col == "std"]
    for j in std_cols:
        for i in range(len(df.index) + 1):
            cell = table[(i, j)]
            cell.set_width(0.08)

    # Apply cell colors for max/min values
    for i, row in enumerate(df.itertuples()):
        for j, val in enumerate(row[1:]):
            cell = table[(i + 1, j)]
            if df.columns[j] in ['avg_overlap', 'SR', 'avg_SR']:
                if val == df[df.columns[j]].max():
                    cell.set_facecolor('#a1d99b')
            if df.columns[j] in ['avg_cle', 'avg_time', "RC", "avg_RC"]:
                if val == df[df.columns[j]].min():
                    cell.set_facecolor('#a1d99b')

    plt.savefig(output_file_img, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)
    print(f'Results table saved as an image to {output_file_img}')


def plot_bar_chart(df, metric, output_file_img):
    metrics_full_names = {
        'avg_overlap': 'Average Overlap',
        'avg_cle': 'Average CLE',
        'avg_time': 'Average Processing Time',
        'SR': 'Average Success Rate',
        'RC': 'Average Reinit Count'
    }

    unit = {
        'avg_overlap': '',
        'avg_cle': '[px]',
        'avg_time': '[s]',
        'SR': '',
        'RC': ''
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y=metric, data=df, ci=None)
    plt.title(f'{metrics_full_names[metric]} results for each tracker')
    plt.ylabel(metrics_full_names[metric] + f' {unit[metric]}')
    plt.xlabel('Trackers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file_img, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()
    print(f'Bar plot saved as an image to {output_file_img}')


def main(base_dir):
    config_file = os.path.join(base_dir, 'config.yaml')
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    reinit_strategy = config.get('reinit_strategy', '')
    exclude_columns = ['RC', 'RC_std'] if reinit_strategy == 'one_init' else []

    overall_results = {
        'CSRT': {'avg_overlap': [], 'avg_overlap_std': [], 'avg_cle': [], 'avg_cle_std': [], 'avg_time': [], 'avg_time_std': [], 'SR': [], 'RC': []},
        'VIT': {'avg_overlap': [], 'avg_overlap_std': [], 'avg_cle': [], 'avg_cle_std': [], 'avg_time': [], 'avg_time_std': [], 'SR': [], 'RC': []},
        'ModVIT': {'avg_overlap': [], 'avg_overlap_std': [], 'avg_cle': [], 'avg_cle_std': [], 'avg_time': [], 'avg_time_std': [], 'SR': [], 'RC': []},
        'DaSiam': {'avg_overlap': [], 'avg_overlap_std': [], 'avg_cle': [], 'avg_cle_std': [], 'avg_time': [], 'avg_time_std': [], 'SR': [], 'RC': []},
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
            subfolder_results = round_results(subfolder_results)
            subfolder_df = pd.DataFrame(subfolder_results).T
            print(f"Results for {root}:\n{subfolder_df}\n")
            subfolder_name = os.path.basename(root)
            subfolder_output_file_img = os.path.join(plots_dir, f'{subfolder_name}_average_results.png')
            save_table_as_image(subfolder_df, subfolder_output_file_img, exclude_columns)

            # Append current subfolder results to the overall results
            for tracker in overall_results.keys():
                for key in overall_results[tracker].keys():
                    overall_results[tracker][key].append(data[tracker][key])

    overall_averages = calculate_overall_averages_and_std(overall_results)
    overall_df = pd.DataFrame(overall_averages).T
    print(f"Overall Results:\n{overall_df}\n")
    overall_output_file_img = os.path.join(plots_dir, 'overall_average_tracker_results.png')
    save_table_as_image(overall_df, overall_output_file_img, exclude_columns, True)

    bar_plots_dir = os.path.join(plots_dir, 'bar_plots')
    os.makedirs(bar_plots_dir, exist_ok=True)

    # Generate bar plots for each metric
    for metric in ['avg_overlap', 'avg_cle', 'avg_time', 'SR', 'RC']:
        bar_plot_output_file_img = os.path.join(bar_plots_dir, f'bars_{metric}.png')
        plot_bar_chart(overall_df, metric, bar_plot_output_file_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML files and compare tracker results.")
    parser.add_argument('base_dir', type=str, help="Base directory to scan for summary.yaml files.")
    args = parser.parse_args()
    main(args.base_dir)
