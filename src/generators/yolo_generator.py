import os
import cv2
import glob
import yaml
import shutil
from collections import Counter 
from ultralytics import YOLO

from src.analyzers.yolo_analyzer import YoloDatasetAnalyzer

class YoloDatasetGenerator:
    """
    Enhances a YOLO dataset by adding new annotated samples from raw data.
    Skips adding bounding boxes if it breaks the class balance threshold.
    """

    def __init__(self, data_yaml_path, weights_path, raw_data_path, output_path, balance_threshold=0.1):
        self.data_yaml_path = data_yaml_path
        self.weights_path = weights_path
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.balance_threshold = balance_threshold

        # Initialize YOLO model
        self.model = YOLO(self.weights_path)

        # Analyze the existing dataset
        self.analyzer = YoloDatasetAnalyzer(self.data_yaml_path)
        self.analyzer.analyze_dataset()

        # We'll reuse the class names + train counts from the existing dataset
        self.class_names = self.analyzer.class_names
        self.train_class_counts = Counter(self.analyzer.train_class_counts)

    def generate_dataset(self):
        """
        Main entry point: processes raw data and updates dataset + data.yaml.
        """
        # 1. Process images or video
        if os.path.isdir(self.raw_data_path):
            self._process_images(self.raw_data_path)
        elif os.path.isfile(self.raw_data_path) and self.raw_data_path.endswith('.mp4'):
            self._process_video(self.raw_data_path)
        else:
            raise ValueError("Invalid raw data path. Provide a folder with images or an MP4 file.")

        # 2. Update YAML file to reference new images
        self._generate_updated_yaml()

    def _process_images(self, folder_path):
        # Capture multiple image types
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        self._process_files(image_files)

    def _process_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        frame_dir = os.path.join(self.output_path, 'video_frames')
        os.makedirs(frame_dir, exist_ok=True)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frame_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()

        # After extracting frames, treat them like images
        self._process_images(frame_dir)

    def _process_files(self, files):
        annotations_dir = os.path.join(self.output_path, 'labels')
        images_dir = os.path.join(self.output_path, 'images')
        os.makedirs(annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        for file_path in files:
            image = cv2.imread(file_path)
            results = self.model.predict(source=image, save=False, save_txt=False)

            if not results:
                # If model didn't return results, skip
                continue

            # Each result is for one image, so typically length = 1
            for result in results:
                if len(result.boxes) == 0:
                    continue  # No detections

                # Prepare label path and copy image
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                label_path = os.path.join(annotations_dir, base_name + '.txt')
                shutil.copy(file_path, os.path.join(images_dir, os.path.basename(file_path)))

                with open(label_path, 'w') as label_file:
                    for box, cls_ in zip(result.boxes.xyxy, result.boxes.cls):
                        x_min, y_min, x_max, y_max = box
                        class_id = int(cls_)

                        # Check if adding this detection is within the balance threshold
                        if not self._is_balanced(class_id):
                            continue

                        
                        # Ensure it matches your usage or YOLO format standard
                        label_file.write(f"{class_id} {x_min} {y_min} {x_max} {y_max}\n")
                        self.train_class_counts[class_id] += 1

    def _is_balanced(self, class_id):
        total_instances = sum(self.train_class_counts.values())
        if total_instances == 0:
            return True
        class_instances = self.train_class_counts[class_id]
        current_ratio = class_instances / total_instances
        return (current_ratio < self.balance_threshold)

    def _generate_updated_yaml(self):
        updated_yaml = {
            "names": self.class_names,
            "nc": len(self.class_names),
            "train": os.path.join(self.output_path, "images"),
            "val": self.analyzer.data_config.get("val", ""),  
        }

        os.makedirs(self.output_path, exist_ok=True)
        yaml_path = os.path.join(self.output_path, 'data.yaml')

        with open(yaml_path, 'w') as file:
            yaml.dump(updated_yaml, file)

        print(f"[INFO] Updated dataset configuration saved to {yaml_path}")
