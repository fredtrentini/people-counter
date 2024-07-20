import argparse
import json
import os

from config import (
    DATASET_ANNOTATIONS_FOLDER
)
from dataset import Dataset
from utils import setup

def merge(data: dict, metadata: dict) -> dict:
    merged_data = {
        "info": metadata["info"],
        "images": metadata["images"],
        "annotations": data["annotations"],
        "categories": metadata["categories"],
    }

    return merged_data

def main():
    setup()

    parser = argparse.ArgumentParser(description="Merges 2 annotation json files, preserving the metadata in one and labels in the other")
    parser.add_argument(
        "-metadata", 
        type=str, 
        required=True, 
        help="File with correct metadata"
    )
    parser.add_argument(
        "-data", 
        type=str, 
        required=True, 
        help="File with correct data"
    )
    args = parser.parse_args()
    metadata_filename = args.metadata
    data_filename = args.data

    metadata_filename = os.path.join(DATASET_ANNOTATIONS_FOLDER, metadata_filename)
    assert os.path.exists(metadata_filename), "Metadata filename not found"

    data_filename = os.path.join(DATASET_ANNOTATIONS_FOLDER, data_filename)
    assert os.path.exists(data_filename), "Data filename not found"
    
    with open(metadata_filename) as file:
        metadata = json.load(file)
    
    with open(data_filename) as file:
        data = json.load(file)
    
    print("Merging cvat annotations...")
    merged_data = merge(data, metadata)
    
    with open(Dataset.get_annotations_path(), "w") as file:
        json.dump(merged_data, file, indent=4)
    
    print(f"CVAT annotations succesfully written to {Dataset.get_annotations_path()}")

if __name__ == "__main__":
    main()
