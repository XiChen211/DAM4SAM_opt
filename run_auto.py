import os
import glob
import argparse
import shutil
import sys
import numpy as np
import cv2
from PIL import Image
path_to_sam2_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_sam2_submodule)
from dam4sam_tracker import DAM4SAMTracker
import time

# SAM2_VERSION_TRACK=dam4sam python run_auto.py --dir frames-dir --ext jpg --output_dir output-dir --model_path sam2_opt/sam2/checkpoints/opts
def run_sequence(dir_path, file_extension, output_dir, model_path=None):
    if not output_dir:
        print("Please provide a path to save the output masks")
        return

    start_time_total = time.time() 

    frames_dir = sorted(glob.glob(os.path.join(dir_path, '*.%s' % file_extension)))
    if not frames_dir:
        print('Error: No image files found in the specified directory.')
        return

    # Define three prompt options
    # point
    prompt_option_1 = {
        "pos_points": [(331, 187)], "neg_points": [], "box": None
    }
    # box
    prompt_option_2 = {
        "pos_points": [], "neg_points": [], "box": (636, 213, 298, 471)
    }
    # box + neg point
    prompt_option_3 = {
        "pos_points": [], "neg_points": [(827, 284)], "box": (636, 213, 298, 471)
    }
    init_prompts = prompt_option_3
    print(f"Using hard-coded prompts: {init_prompts}")

    if not init_prompts:
        print('Error: No valid initial annotations (points or box) provided.')
        return
    
    if not model_path:
        # torch
        tracker = DAM4SAMTracker(tracker_name="sam21pp-L")
    else:
        # speedup with tensorrt
        tracker = DAM4SAMTracker(tracker_name="sam21pp-L", backend="tensorrt", model_root_path=model_path)

    
    # Setup output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Starting to segment the frame sequence...')

    start_time_loop = time.time()
    
    for i, frame_path in enumerate(frames_dir):
        img = Image.open(frame_path).convert('RGB')

        if i == 0:
            # Initialize the tracker on the first frame
            outputs = tracker.initialize(img, init_prompts=init_prompts)
        else:
            # Track the object in subsequent frames
            outputs = tracker.track(img)

        pred_mask = outputs['pred_mask']

        # Save the resulting mask to the output directory
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        output_path = os.path.join(output_dir, f'{frame_name}.png')
        cv2.imwrite(output_path, pred_mask * 255)

    end_time_loop = time.time()
    loop_duration = end_time_loop - start_time_loop
    num_frames = len(frames_dir)
    
    print('Segmentation finished.')
    print("\n" + "="*50)
    print("Performance Analysis Results:")
    print("="*50)
    if num_frames > 0:
        avg_time_per_frame = loop_duration / num_frames
        fps = num_frames / loop_duration
        print(f"Core processing loop duration: {loop_duration:.2f} seconds")
        print(f"Total frames processed: {num_frames} frames")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
        print(f"Processing speed (FPS): {fps:.2f} frames/sec")
    else:
        print("No frames were processed.")
    print("="*50)
    
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"\nTotal function execution time: {total_duration:.2f} seconds (including model loading, file I/O, etc.)")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run on a sequence of frames-dir.')
    parser.add_argument('--dir', type=str, required=True, help='Path to directory with frames-dir.')
    parser.add_argument('--ext', type=str, default='jpg', help='Image file extension.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory. Required for non-GUI execution.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the output directory. Required for non-GUI execution.')
    args = parser.parse_args()
    run_sequence(args.dir, args.ext, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()