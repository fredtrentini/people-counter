# People Counter

This is a project to make a CNN able to accurately count people in videos.

## Steps

### 1- Read raw video data to create image dataset

Input: `./data/[*video_name.avi|mp4]`

Output: `./dataset/[*video_name]/[*frame.jpg]`

### 2- Read image dataset to generate annotations

Automatically generated annotations are not perfect, consider manually optimizing them using [cvat](https://app.cvat.ai) and re-running next steps. (COCO 1.0 format is used to export/import annotations)

Input: `./dataset/[*video_name]/[*frame.jpg]`

Output: `./dataset-annotations/[*video_name]/[*frame_annotation.json]`

### 3- Using the image dataset and the associated annotations, build a custom image model on top of the pretrained model to optimize detection for the dataset images domain

### 4- Test people detection in real time reading the current video device

## Requirements

```bash
pip install -r requirements.txt
```

## Scripts

- `plot.py`: Visualize current existing annotations
  - Generate dataset and annotations if they don't exist
  - Create desktop windows to visualize the annotations
  - About the data source for the annotations:
    - By default, annotations come from model predictions file
    - If `-file` argument is used, parse a given file as cvat annotations instead
- `merge_cvat_annotations.py`: When importing annotations from cvat, a lot of information is either lost or overwritten such as category_id, info, categories, etc, so this script solves this problem by inputting 2 annotation files (the one which got exported and the one who just got imported) and correctly merging the information
  - Input two relative filenames, which are expected to be the file with the correct metadata and the file with the correct labels
  - Write new merged file to the current file path (overwrites existing file)
- `main.py`: Main project file which does the following
  - Generate dataset and annotations if they don't exist
  - Train the pretrained model (TODO)
  - Test the application in real-time using the current video device (TODO)

## Practical overview (How to use those scripts)

- First of all download some video files and move them to `data` folder.
- Run `plot.py` to build dataset, build dataset annotations and generate initial predictions, as well as visualize those predictions.
- Create a task on cvat and import the annotations to visualize the predicted bounding boxes and manually optimize them as much as needed.
- Download the updated annotations and run `merge_cvat_annotations.py` passing the original annotations and the downloaded annotations as arguments.
