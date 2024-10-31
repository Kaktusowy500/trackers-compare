import os
import csv
import argparse
import matplotlib.pyplot as plt


def process_tracker_results_from_directory(directory, frame_limit=None):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Create a plots directory within the provided directory
    plots_directory = os.path.join(directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    tracker_data = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith("results.csv"):
            filepath = os.path.join(directory, filename)
            tracker_name = os.path.splitext(filename)[0]

            frame_numbers = []
            overlaps = []
            errors = []
            processing_times = []
            bbox_areas = []
            valid_indicators = []

            with open(filepath, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if not row['Frame'].isdigit():
                        continue

                    frame_numbers.append(int(row['Frame']))
                    overlaps.append(float(row['Overlap']))
                    errors.append(float(row['Center Error']))
                    processing_times.append(float(row['Processing Time']))
                    bbox_areas.append(float(row['BBox Area']))
                    valid_indicators.append(float(row['Valid']))

            tracker_data[tracker_name] = {
                'frames': frame_numbers,
                'overlaps': overlaps,
                'errors': errors,
                'processing_times': processing_times,
                'bbox_areas': bbox_areas,
                'valid_indicators': valid_indicators
            }

       # Determine the maximum frame number to plot
    if frame_limit is not None:
        frame_limit = min(frame_limit, *(max(data['frames']) for data in tracker_data.values()))
    else:
        frame_limit = max(max(data['frames']) for data in tracker_data.values())

    # Function to filter data by frame limit
    def filter_data_by_frame_limit(data, limit):
        return [x for f, x in zip(data['frames'], data['overlaps']) if f <= limit], \
               [x for f, x in zip(data['frames'], data['errors']) if f <= limit], \
               [x for f, x in zip(data['frames'], data['processing_times']) if f <= limit], \
               [x for f, x in zip(data['frames'], data['bbox_areas']) if f <= limit]

    # Plot Overlaps
    plt.figure(figsize=(10, 6))
    for tracker_name, data in tracker_data.items():
        filtered_overlaps, _, _, _ = filter_data_by_frame_limit(data, frame_limit)
        filtered_frames = [f for f in data['frames'] if f <= frame_limit]
        plt.plot(filtered_frames, filtered_overlaps, label=f"{tracker_name} Overlap")
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
        _, filtered_errors, _, _ = filter_data_by_frame_limit(data, frame_limit)
        filtered_frames = [f for f in data['frames'] if f <= frame_limit]
        plt.plot(filtered_frames, filtered_errors, label=f"{tracker_name} Center Error")
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
        _, _, filtered_processing_times, _ = filter_data_by_frame_limit(data, frame_limit)
        filtered_frames = [f for f in data['frames'] if f <= frame_limit]
        plt.plot(filtered_frames, filtered_processing_times, label=f"{tracker_name} Processing Time")
    plt.xlabel('Frame Number')
    plt.ylabel('Processing Time (s)')
    plt.title('Processing Time per Frame for All Trackers')
    plt.grid(True)
    plt.legend()
    processing_time_plot_path = os.path.join(plots_directory, "processing_time_plot.png")
    plt.savefig(processing_time_plot_path)
    plt.close()

    # Plot BBox Areas
    plt.figure(figsize=(10, 6))
    for tracker_name, data in tracker_data.items():
        _, _, _, filtered_bbox_areas = filter_data_by_frame_limit(data, frame_limit)
        filtered_frames = [f for f in data['frames'] if f <= frame_limit]
        plt.plot(filtered_frames, filtered_bbox_areas, label=f"{tracker_name} BBox Area")
    plt.xlabel('Frame Number')
    plt.ylabel('Bounding Box Area (px)')
    plt.title('Bounding Box area per Frame for All Trackers')
    plt.grid(True)
    plt.legend()
    bbox_area_plot_path = os.path.join(plots_directory, "bbox_area_plot.png")
    plt.savefig(bbox_area_plot_path)
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
    parser.add_argument('--frame_limit', type=int, help="Optional frame limit to plot up to.")
    args = parser.parse_args()

    process_tracker_results_from_directory(args.directory, args.frame_limit)


if __name__ == "__main__":
    main()
