import os
import csv
import argparse
import matplotlib.pyplot as plt


def process_tracker_results_from_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Create a plots directory within the provided directory
    plots_directory = os.path.join(directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    tracker_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            tracker_name = os.path.splitext(filename)[0]

            frame_numbers = []
            overlaps = []
            errors = []
            processing_times = []

            with open(filepath, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if not row['Frame'].isdigit():
                        continue

                    frame_numbers.append(int(row['Frame']))
                    overlaps.append(float(row['Overlap']))
                    errors.append(float(row['Center Error']))
                    processing_times.append(float(row['Processing Time']))

            tracker_data[tracker_name] = {
                'frames': frame_numbers,
                'overlaps': overlaps,
                'errors': errors,
                'processing_times': processing_times
            }

    # Plot Overlaps
    plt.figure(figsize=(10, 6))
    for tracker_name, data in tracker_data.items():
        plt.plot(data['frames'], data['overlaps'], label=f"{tracker_name} Overlap")
    plt.xlabel('Frame Number')
    plt.ylabel('Overlap')
    plt.title('Overlap per Frame for All Trackers')
    plt.grid(True)
    plt.legend()
    overlap_plot_path = os.path.join(plots_directory, "overlap_plot.png")
    plt.savefig(overlap_plot_path)
    plt.close()

    # Plot Center Errors
    plt.figure(figsize=(10, 6))
    for tracker_name, data in tracker_data.items():
        plt.plot(data['frames'], data['errors'], label=f"{tracker_name} Center Error")
    plt.xlabel('Frame Number')
    plt.ylabel('Center Error')
    plt.title('Center Error per Frame for All Trackers')
    plt.grid(True)
    plt.legend()
    error_plot_path = os.path.join(plots_directory, "center_error_plot.png")
    plt.savefig(error_plot_path)
    plt.close()

    # Plot Processing Times
    plt.figure(figsize=(10, 6))
    for tracker_name, data in tracker_data.items():
        plt.plot(data['frames'], data['processing_times'], label=f"{tracker_name} Processing Time")
    plt.xlabel('Frame Number')
    plt.ylabel('Processing Time (s)')
    plt.title('Processing Time per Frame for All Trackers')
    plt.grid(True)
    plt.legend()
    processing_time_plot_path = os.path.join(plots_directory, "processing_time_plot.png")
    plt.savefig(processing_time_plot_path)
    plt.close()

    # Print Average Values for Each Tracker
    for tracker_name, data in tracker_data.items():
        avg_overlap = sum(data['overlaps']) / len(data['overlaps'])
        avg_error = sum(data['errors']) / len(data['errors'])
        avg_processing_time = sum(data['processing_times']) / len(data['processing_times'])
        print(f"Tracker: {tracker_name}")
        print(f"  Average Overlap: {avg_overlap}")
        print(f"  Average Center Error: {avg_error}")
        print(f"  Average Processing Time: {avg_processing_time}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Process and visualize tracker performance results.")
    parser.add_argument('directory', type=str, help="Path to the directory containing the results files.")
    args = parser.parse_args()

    process_tracker_results_from_directory(args.directory)


if __name__ == "__main__":
    main()
