import os
import cv2
import argparse


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps else None

    cap.release()
    return frames, duration


def analyze_dataset(dataset_path):
    total_frames = 0
    total_duration = 0
    durations = []
    frame_counts = []
    video_count = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                frames, duration = get_video_info(video_path)
                if frames is not None and duration is not None:
                    total_frames += frames
                    total_duration += duration
                    durations.append(duration)
                    frame_counts.append(frames)
                    video_count += 1
                    print(f"Processed {video_path}: {frames} frames, {duration:.2f} seconds")

    if durations:
        min_duration = min(durations)
        max_duration = max(durations)
        avg_duration = total_duration / video_count if video_count else 0
        min_frames = min(frame_counts)
        max_frames = max(frame_counts)
    else:
        min_duration = max_duration = avg_duration = 0
        min_frames = max_frames = 0

    print("\nSummary:")
    print(f"Total number of videos: {video_count}")
    print(f"Total frames: {total_frames}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average duration: {avg_duration:.2f} seconds")
    print(f"Minimum duration: {min_duration:.2f} seconds")
    print(f"Maximum duration: {max_duration:.2f} seconds")
    print(f"Minimum number of frames: {min_frames}")
    print(f"Maximum number of frames: {max_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the video dataset directory")
    args = parser.parse_args()

    analyze_dataset(args.dataset_path)
