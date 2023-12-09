"""A script to augment and resize a dataset of images. Accepts values in the PASCAL VOC format and returns an augmented data set in the MS COCO format.
"""
import xml.etree.ElementTree as ET
import os
import cv2
import json
import matplotlib
import albumentations as A
from datetime import datetime
import tempfile
import webbrowser

matplotlib.use("TkAgg")  # Use the Tkinter backend


def parse_pascal_voc(filename: str) -> dict:
    """
    Helper function which parses Pascal VOC XML file, giving a dictionary of the info

    Args:
        filename (str): Path to the XML file.

    Returns:
    - image_info (dict): Dictionary containing image information.
    """

    tree = ET.parse(filename)  # Parse the element tree and get the root
    root = tree.getroot()

    image_info = {
        "filename": root.find("filename").text,  # Parse to extract a list of filenames
        "size": {  # Parse to get the sizes of th eimage
            "width": int(root.find("size/width").text),
            "height": int(root.find("size/height").text),
            "depth": int(root.find("size/depth").text),
        },
        "objects": [],
    }

    for obj in root.findall("object"):
        obj_info = {
            "name": obj.find("name").text,  # Extract the catergory
            "bbox": {  # Extract the bbox settings
                "xmin": int(obj.find("bndbox/xmin").text),
                "ymin": int(obj.find("bndbox/ymin").text),
                "xmax": int(obj.find("bndbox/xmax").text),
                "ymax": int(obj.find("bndbox/ymax").text),
            },
        }
        image_info["objects"].append(
            obj_info
        )  # Collect all the bbox objects into a list

    return image_info


def read_pascal_voc_images_in_dir(data_dir: str) -> dict:
    """Reads through all annoatations and images in a directory, parsing them from Pascal VOC

    Args:
        data_dir (str): Directory of the dataset

    Returns:
        dict: A list of dictionaries that contain the images and bboxes
    """

    data = []

    images_dir = os.path.join(
        data_dir, "images"
    )  # Assume the images are in a folder called images inside the root folder
    annotations_dir = os.path.join(
        data_dir, "annotations"
    )  # Assume the annotations are in a folder called annotations inside the root folder

    # Some images are 415x416 some are 416x416. Resize to ensure it's standardrized
    # For this one, we are not changing the Bboxes, but since we are rescale by a fraction of a precent, there is no effect
    transform = A.Compose(
        [
            A.Resize(width=416, height=416),  # Adjust width and height as needed
        ]
    )  # Define an Albumentations resize transform.

    for filename in os.listdir(annotations_dir):  # Search through all the XML files
        if filename.endswith(".xml"):
            # Extract the image info for a given image
            xml_path = os.path.join(annotations_dir, filename)
            image_info = parse_pascal_voc(xml_path)

            # Load image
            image_path = os.path.join(images_dir, image_info["filename"])
            image = cv2.imread(image_path)

            # Resize the images using Albumentations and put the resized image in the dictionary
            transformed = transform(image=image)
            resized_image = transformed["image"]
            image_info["image"] = resized_image

            data.append(image_info)

    return data


def visualize_pascal_voc_image(image, objects):
    """
    Debugging Function to visualize the image with bounding boxes and object names.
    This does draw the bounding boxes onto the final images which is done.
    This is needed due to weird rendering issues with CV2 on some Linux systems (developed on Linux Mint).
    DO NOT ENABLE WHEN CREATING THE FINAL DATASET

    Args:
        - image (numpy.ndarray): The image to be visualized.
        - objects (list): List of dictionaries containing object (bbox) information.
    """
    for obj in objects:  # For all the objects (bboxes)
        bbox = obj["bbox"]
        cv2.rectangle(
            image,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            (0, 255, 0),
            2,
        )  # Draw a rectangle using the bbox properties in pascal voc format
        cv2.putText(
            image,
            obj["name"],
            (bbox["xmin"], bbox["ymin"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )  # Add the category text

    # Save the image to a temporary file
    temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    cv2.imwrite(temp_image_path, image)

    # Open the image with the default image viewer
    webbrowser.open(temp_image_path)

    # Close the temporary image file
    os.remove(temp_image_path)


def augmentImage(image, objects):
    """
    Applys all the augmentations as specified in the report to the image.

    Args:
        image (numpy.ndarray): Input image.
        objects (list): List of dictionaries, each containing 'name' and 'bbox' keys.

    Returns:
        augmented_image (numpy.ndarray): Image after flip augmentation.
        augmented_objects (list): Updated objects after flip augmentation.
    """

    # Define an Albumentations transform with all the possible augmentations
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),  # Add a horizontal flip with 50% probablililty
            A.RandomSizedCrop(
                p=1.0, min_max_height=[354, 354], w2h_ratio=1, height=416, width=416
            ),  # Crop that maintains the aspect ratio, and rescale back to 416 by 416. Random crop ratio, 0-15%, random centering.
            A.ColorJitter(
                p=1.0, saturation=0.25, brightness=0.15, contrast=0.15
            ),  # Random brightness, contrast, and satruation. +- 15%, +-15%, +-25% respectively, randomized.
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # Apply the transformation
    augmented = transform(
        image=image,
        bboxes=[list(obj["bbox"].values()) for obj in objects],
        class_labels=[obj["name"] for obj in objects],
        format="pascal_voc",
    )
    augmented_image = augmented["image"]

    # Convert the updated objects back to the original format and package it up back in the dict
    augmented_objects = [
        {
            "name": obj["name"],
            "bbox": {
                "xmin": int(box[0]),
                "ymin": int(box[1]),
                "xmax": int(box[2]),
                "ymax": int(box[3]),
            },
        }
        for obj, box in zip(objects, augmented["bboxes"])
    ]

    return augmented_image, augmented_objects


def export_pascal_voc_to_coco(pascal_voc_data, output_dir):
    """
    Convert data formated in Pascal VOC from dict to COCO format and export to a JSON file.

    Args:
        pascal_voc_data (list): List of dictionaries containing image information in Pascal VOC format.
        output_dir (str): Directory to save the resulting COCO-format JSON annotation file and images.
    """

    coco_data = {
        "info": {
            "year": "2023",
            "version": "1",
            "description": "Generated for CISC867",
            "contributor": "",
            "date_created": datetime.today().strftime("%Y-%m-%d"),
        },  # Use some generic information
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0",
            }  # Use the creative commons lisence although it's unknown how these images were originally licensed whey they were put on Kaggle
        ],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    category_mapping = {
        "head": 1,
        "helmet": 2,
        "person": 3,
    }  # A dict that maps Pascal VOC class names to COCO category IDs

    os.makedirs(
        output_dir, exist_ok=True
    )  # Create output directory if it doesn't exist

    # Populate categories in COCO format
    for idx, class_name in enumerate(category_mapping.keys(), start=1):
        coco_data["categories"].append(
            {
                "id": idx,
                "name": class_name,
                "supercategory": "object",
            }
        )

    # Start the ID counters at 0. For some reason, MMdetection requires it to start at 0, instead of 1 that some other libraries accept
    image_id_counter = 0
    annotation_id_counter = 0

    # Populate images and annotations in COCO format
    for image_info in pascal_voc_data:
        image = image_info["image"]
        objects = image_info["objects"]

        height, width, _ = image.shape  # Get height and width from the image data

        # Save the modified image as a new png
        output_image_path = os.path.join(output_dir, f"{image_info['filename']}")
        cv2.imwrite(output_image_path, image)

        # Add image information to COCO data
        coco_data["images"].append(
            {
                "id": image_id_counter,
                "license": 1,
                "file_name": image_info["filename"],
                "height": height,
                "width": width,
                "date_captured": datetime.today().strftime(
                    "%Y-%m-%d"
                ),  # The date captured is just set to the current date
            }
        )

        # Add object annotations to COCO data
        for obj in objects:
            bbox = obj["bbox"]

            coco_data["annotations"].append(
                {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": category_mapping[obj["name"]],
                    "segmentation": [],  # You can fill this with segmentation information if available
                    "area": (bbox["xmax"] - bbox["xmin"])
                    * (bbox["ymax"] - bbox["ymin"]),
                    "bbox": [
                        bbox["xmin"],
                        bbox["ymin"],
                        bbox["xmax"] - bbox["xmin"],
                        bbox["ymax"] - bbox["ymin"],
                    ],  # This conversion is need since COCO uses xmin, ymin, width, height instead of xy mins.
                    "iscrowd": 0,
                }
            )

            annotation_id_counter += 1

        image_id_counter += 1

    # Save the COCO-format JSON file
    output_json_path = os.path.join(output_dir, "coco_annotations.json")
    with open(output_json_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=2)


if __name__ == "__main__":
    parsed_data = read_pascal_voc_images_in_dir("helmetSafetyTestSet")

    for image_info in parsed_data:
        image_info["image"], image_info["objects"] = augmentImage(
            image_info["image"], image_info["objects"]
        )

        # visualize_pascal_voc_image(
        #     image_info["image"], image_info["objects"]
        # )  # Enable this to visualize an image. DO NO ENABLE WHEN RUNNING FINAL AUGMENTATION!!!

    export_pascal_voc_to_coco(parsed_data, "test")
