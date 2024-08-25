import os
import xml.etree.ElementTree as ET
import argparse


def convert_xml_to_txt(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get original image dimensions
    original_width = int(root.find('.//original_size/width').text)
    original_height = int(root.find('.//original_size/height').text)

    # Initialize a dictionary to store annotations by frame
    annotations = {}
    for box in root.findall('.//box'):
        frame = int(box.get('frame'))
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))
        occluded = box.get('occluded')

        # Normalize coordinates
        normalized_x = (xtl + (xbr - xtl) / 2) / original_width
        normalized_y = (ytl + (ybr - ytl) / 2) / original_height
        normalized_width = (xbr - xtl) / original_width
        normalized_height = (ybr - ytl) / original_height
        oc_value = int(occluded) if occluded.isdigit() else 0

        annotations[frame] = (normalized_x, normalized_y, normalized_width, normalized_height, oc_value)

    # Determine the total number of frames
    meta = root.find('.//meta/job')
    total_frames = int(meta.find('size').text)

    with open(output_file, 'w') as f:
        for frame in range(total_frames):
            if frame in annotations:
                normalized_x, normalized_y, normalized_width, normalized_height, oc_value = annotations[frame]
                f.write(f"{frame},{normalized_x:.6f},{normalized_y:.6f},{normalized_width:.6f},{normalized_height:.6f},{oc_value}\n")
            else:
                f.write(f"{frame}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to TXT format.")
    parser.add_argument('input_file', type=str, help="Path to the input XML file.")
    parser.add_argument('output_file', type=str, help="Path to the output TXT file.")
    args = parser.parse_args()

    convert_xml_to_txt(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
