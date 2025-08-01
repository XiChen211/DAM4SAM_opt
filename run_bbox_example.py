import os
import sys
# Add the path to the sam2 submodule to the system path
path_to_sam2_submodule = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_sam2_submodule)
import glob
import argparse
import shutil

import numpy as np
import cv2
from PIL import Image

from dam4sam_tracker import DAM4SAMTracker
from utils.visualization_utils import overlay_mask, overlay_rectangle

# python run_bbox_example.py --dir frames-dir --ext jpg --output_dir output-dir
class InputSelector:
    """
    A unified input selector that supports mixed annotations:
    - Left-click: Add a foreground point (green)
    - Right-click: Add a background point (red)
    - Left-drag: Draw a bounding box (blue)
    - 'c' key: Clear all current annotations
    - 'Enter' key: Confirm and proceed
    """

    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.display_image = image.copy()

        # Store all types of prompts
        self.prompts = {
            "pos_points": [],
            "neg_points": [],
            "box": None
        }

        self.start_point = None
        self.is_drawing = False

    def _mouse_callback(self, event, x, y, flags, param):
        # Left button down, start drawing or marking a point
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_drawing = True

        # Mouse move, if drawing, display the bounding box in real-time
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                # Create a temporary image to show the dragging process, avoiding leaving marks on the original display
                temp_image = self.display_image.copy()
                cv2.rectangle(temp_image, self.start_point, (x, y), (255, 0, 0), 2)
                cv2.imshow(self.window_name, temp_image)

        # Left button up, finish drawing or marking a point
        elif event == cv2.EVENT_LBUTTONUP:
            end_point = (x, y)
            self.is_drawing = False

            # If the start and end points are very close, treat it as a click
            if np.linalg.norm(np.array(self.start_point) - np.array(end_point)) < 5:
                self.prompts["pos_points"].append(end_point)
                print(f"Added foreground point: {end_point}")
                cv2.circle(self.display_image, end_point, 5, (0, 255, 0), -1)
            # Otherwise, treat it as a drag to form a bounding box
            else:
                x1, y1 = self.start_point
                x2, y2 = end_point
                # If a box already exists, the new one will replace it
                self.prompts["box"] = (min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))
                print(f"Set bounding box: {self.prompts['box']}")
                # Clear old drawings and redraw all prompts
                self._redraw_prompts()

            cv2.imshow(self.window_name, self.display_image)

        # Right-click, add a background point
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.prompts["neg_points"].append((x, y))
            print(f"Added background point: {(x, y)}")
            cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, self.display_image)

    def _redraw_prompts(self):
        """Redraw all current prompts (points and box) for updating or clearing."""
        self.display_image = self.image.copy()
        # Draw box
        if self.prompts["box"]:
            x, y, w, h = self.prompts["box"]
            cv2.rectangle(self.display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Draw foreground points
        for pt in self.prompts["pos_points"]:
            cv2.circle(self.display_image, pt, 5, (0, 255, 0), -1)
        # Draw background points
        for pt in self.prompts["neg_points"]:
            cv2.circle(self.display_image, pt, 5, (0, 0, 255), -1)

    def select_input(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n--- Mixed Annotation Mode ---")
        print("Left-click: Add a foreground point (green)")
        print("Right-click: Add a background point (red)")
        print("Left-drag: Draw/Replace bounding box (blue)")
        print("Press 'c' to clear all annotations")
        print("Press 'Enter' to confirm and start tracking")
        print("--------------------\n")

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("Clearing all annotations.")
                self._reset_selection()
                self._redraw_prompts()

            elif key == 13:  # Enter key
                break

        cv2.destroyWindow(self.window_name)

        # Check if there is at least one valid prompt
        if not self.prompts["pos_points"] and not self.prompts["neg_points"] and not self.prompts["box"]:
            return None

        return self.prompts

    def _reset_selection(self):
        self.prompts = {
            "pos_points": [],
            "neg_points": [],
            "box": None
        }
        self.start_point = None
        self.is_drawing = False


def run_sequence(dir_path, file_extension, output_dir):
    frames_dir = sorted(glob.glob(os.path.join(dir_path, '*.%s' % file_extension)))
    if not frames_dir:
        print('Error: No image files found in the specified directory.')
        return

    img0_bgr = cv2.imread(frames_dir[0])

    # Use the new InputSelector
    input_selector = InputSelector('Mixed Annotation Interface', img0_bgr)
    init_prompts = input_selector.select_input()

    if not init_prompts:
        print('Error: No valid initial annotations (points or box) were provided.')
        return

    tracker = DAM4SAMTracker('sam21pp-L')

    if output_dir:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    print('Starting to segment the frame sequence...')
    for i, frame_path in enumerate(frames_dir):
        img = Image.open(frame_path).convert('RGB')
        img_vis = np.array(img)

        if i == 0:
            # Initialize the tracker with a dictionary containing multiple prompts
            outputs = tracker.initialize(img, init_prompts=init_prompts)

            # Visualize all initial prompts
            if not output_dir:
                if init_prompts.get("box"):
                    overlay_rectangle(img_vis, init_prompts["box"], color=(255, 0, 0), line_width=2)
                for point in init_prompts.get("pos_points", []):
                    cv2.circle(img_vis, point, 5, (0, 255, 0), -1)
                for point in init_prompts.get("neg_points", []):
                    cv2.circle(img_vis, point, 5, (0, 0, 255), -1)

            if not output_dir:
                window_name = 'win'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                wait_ = 0
        else:
            outputs = tracker.track(img)

        pred_mask = outputs['pred_mask']

        if output_dir:
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            output_path = os.path.join(output_dir, f'{frame_name}.png')
            cv2.imwrite(output_path, pred_mask * 255)
        else:
            overlay_mask(img_vis, pred_mask, (255, 255, 0), line_width=1, alpha=0.55)
            cv2.imshow(window_name, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
            key_ = cv2.waitKey(wait_)

            if key_ == 27: # ESC key
                exit(0)
            elif key_ == 32: # Space bar
                wait_ = 0 if wait_ else 1

    print('Segmentation complete.')


# main function remains unchanged
def main():
    parser = argparse.ArgumentParser(description='Run on a sequence of frames-dir.')
    parser.add_argument('--dir', type=str, required=True, help='Path to directory with frames-dir.')
    parser.add_argument('--ext', type=str, default='jpg', help='Image file extension.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')

    args = parser.parse_args()
    run_sequence(args.dir, args.ext, args.output_dir)


if __name__ == "__main__":
    main()