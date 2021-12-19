import requests
import os
import glob
import argparse

import torch
from torchvision.io import decode_image
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from PIL import Image, ImageDraw

from src.const import PARAMETERS


class Detector(object):
    """docstring for Detector"""

    def __init__(self, **kwargs):
        super(Detector, self).__init__()
        # Params initiation
        self.model_name = kwargs.get("model_name", "facebook/detr-resnet-50")
        self.threshold = kwargs.get("threshold", 0.99)
        self.max_items = kwargs.get("max_items", 10)
        self.save_destination, self.output_destination = (
            kwargs.get("save_destination", "./test_images"),
            kwargs.get("output_destination", "./output_images"),
        )
        self.max_width, self.max_height = (
            kwargs.get("max_width", 2000),
            kwargs.get("max_height", 2000),
        )
        self.resize, self.resize_scale = (
            kwargs.get("resize", True),
            kwargs.get("resize_scale", 0.75),
        )
        self.excluded_objects = kwargs.get(
            "excluded_objects", [91]
        )  # for COCO dataset that detr is using
        self.image_format = kwargs.get("image_format", "PNG")

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.object_detector = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        )
        self._to_tensor_transformer = transforms.ToTensor()

    def _predict(self, image_path):
        self.image_path = image_path
        self.image_save_name = os.path.basename(self.image_path).split(".")[
            0
        ]  # Get only file name without extension
        if image_path.startswith("http"):
            self.image = Image.open(requests.get(self.image_path, stream=True).raw)
        else:
            self.image = Image.open(self.image_path)

        self._get_image_size()

        if self.resize:
            while self.width > self.max_width or self.height > self.max_height:
                self._resize(self.resize_scale)

        inputs = self.feature_extractor(images=self.image, return_tensors="pt")
        outputs = self.object_detector(**inputs)
        logits, bboxes = outputs.logits, outputs.pred_boxes
        return logits, bboxes

    def _resize(self, scale):
        self.image = self.image.resize(
            (int(self.width * scale), int(self.height * scale))
        )
        self._get_image_size()

    def _get_image_size(self):
        self.width, self.height = self.image.size
        print(f"Width: {self.width}, Height: {self.height}")

    def _get_objects(self, logits, bboxes):
        self.objects = []

        # To show the detected objects
        image_copy = self.image.copy()
        # Background for masking all other items
        self.complementary_image_mask = Image.new(
            "RGB", (self.width, self.height), (0, 0, 0)
        )  # Black
        # Background for masking just this item
        self.this_image_mask = self.complementary_image_mask.copy()

        drw = ImageDraw.Draw(image_copy)
        drw_complementary_image_mask = ImageDraw.Draw(self.complementary_image_mask)

        for index, (logit, box) in enumerate(zip(logits[0], bboxes[0])):
            proba = logit.softmax(-1)  # getting the confidence score
            cls = logit.argmax()  # item index
            if cls in self.excluded_objects or proba[cls] < self.threshold:
                continue  # skip if

            box = box * torch.tensor([self.width, self.height, self.width, self.height])
            x, y, w, h = box

            x0, x1, y0, y1 = x - w // 2, x + w // 2, y - h // 2, y + h // 2

            if len(self.objects) <= self.max_items:
                obj = {}
                obj["index"] = index  # index
                obj["box"] = [(x0, y0), (x1, y1)]  # box bounds
                obj["cls"] = cls  # item index

                self.objects.append(obj)

            drw.rectangle([(x0, y0), (x1, y1)])
            drw.text((x, y), f"{cls}", fill="white")

            drw_complementary_image_mask.rectangle(
                [(x0, y0), (x1, y1)], fill=(255, 255, 255)
            )  # White masking

        image_copy.save(
            f"{self.save_destination}/{self.image_save_name}_detected.{self.image_format.lower()}",
            self.image_format,
        )
        self.image.save(
            f"{self.save_destination}/{self.image_save_name}_complementary.{self.image_format.lower()}",
            self.image_format,
        )
        self.image.save(
            f"{self.save_destination}/{self.image_save_name}_this.{self.image_format.lower()}",
            self.image_format,
        )

    def _masking(self):
        for index, obj in enumerate(self.objects):
            this_image_mask, complementary_image_mask = (
                self.this_image_mask.copy(),
                self.complementary_image_mask.copy(),
            )

            drw_this_image_mask, drw_complementary_image_mask = (
                ImageDraw.Draw(this_image_mask),
                ImageDraw.Draw(complementary_image_mask),
            )
            drw_this_image_mask.rectangle(obj["box"], fill=(255, 255, 255))
            drw_complementary_image_mask.rectangle(obj["box"], fill=(0, 0, 0))

            this_image_mask.save(
                f"{self.save_destination}/{self.image_save_name}_this_mask{index:03d}.{self.image_format.lower()}",
                self.image_format,
            )
            complementary_image_mask.save(
                f"{self.save_destination}/{self.image_save_name}_complementary_mask{index:03d}.{self.image_format.lower()}",
                self.image_format,
            )

    def _create_directory(self):
        for directory in [self.save_destination, self.output_destination]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def process(self, image_path):
        self._create_directory()
        logits, bboxes = self._predict(image_path)
        self._get_objects(logits, bboxes)
        print(f"Detected: {len(self.objects)} objects")
        self._masking()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
    )

    args = parser.parse_args()

    detector = Detector(**PARAMETERS)
    detector.process(args.image_path)
