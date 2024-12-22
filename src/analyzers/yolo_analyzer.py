import os
import yaml
import glob
from collections import Counter

class YoloDatasetAnalyzer:
    """
    Analyzes YOLO-style dataset configurations.
    """

    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        self.data_config = None
        self.train_images_path = None
        self.val_images_path = None
        self.class_names = []
        self.train_class_counts = Counter()
        self.val_class_counts = Counter()
        self.train_samples = 0
        self.val_samples = 0
        self._load_data_config()

    def _load_data_config(self):
        with open(self.data_yaml_path, 'r') as file:
            self.data_config = yaml.safe_load(file)

        base_path = os.path.dirname(self.data_yaml_path)
        self.train_images_path = os.path.join(base_path, self.data_config['train'])
        self.val_images_path   = os.path.join(base_path, self.data_config['val'])
        self.class_names       = self.data_config['names']

    def _count_class_instances_and_samples(self, images_path):
        labels_path = images_path.replace('images', 'labels')
        label_files = glob.glob(os.path.join(labels_path, '*.txt'))

        class_counter = Counter()
        total_samples = len(label_files)
        for label_file in label_files:
            with open(label_file, 'r') as lf:
                for line in lf:
                    class_id = int(line.split()[0])
                    class_counter[class_id] += 1
        return class_counter, total_samples

    def analyze_dataset(self):
        """
        Populates class counts and sample counts for train/val.
        """
        self.train_class_counts, self.train_samples = self._count_class_instances_and_samples(
            self.train_images_path
        )
        self.val_class_counts, self.val_samples = self._count_class_instances_and_samples(
            self.val_images_path
        )

    def get_dataset_statistics(self):
        """
        Returns a dict of basic dataset stats.
        """
        return {
            "total_classes": len(self.class_names),
            "training_samples": self.train_samples,
            "validation_samples": self.val_samples,
            "total_samples": self.train_samples + self.val_samples,
        }

    def get_class_distribution(self):
        """
        Returns class distribution for train/val in terms of absolute counts.
        """
        train_distribution = {
            self.class_names[id_]: count
            for id_, count in self.train_class_counts.items()
        }
        val_distribution = {
            self.class_names[id_]: count
            for id_, count in self.val_class_counts.items()
        }
        return {
            "training": train_distribution,
            "validation": val_distribution,
        }

    def get_class_distribution_percentages(self):
        """
        Returns class distribution for train/val in percentages.
        """
        train_percentages = {}
        val_percentages = {}

        if self.train_samples > 0:
            train_percentages = {
                self.class_names[id_]: (count / self.train_samples) * 100
                for id_, count in self.train_class_counts.items()
            }
        if self.val_samples > 0:
            val_percentages = {
                self.class_names[id_]: (count / self.val_samples) * 100
                for id_, count in self.val_class_counts.items()
            }

        return {
            "training_percentages": train_percentages,
            "validation_percentages": val_percentages,
        }
