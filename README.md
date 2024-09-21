# People Counter

This is a project to make a CNN able to accurately count people in videos.

[Try it out on google colab](https://colab.research.google.com/drive/12H9n7k-PrRxawqukzgTbwYUXutNQpKYx)

## Steps

### 1- Read raw video data to create image dataset

Input: `./data-videos/[*video_name.avi|mp4]`

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

- `test.py`: Visualize current existing annotations
  - Generate dataset and annotations if they don't exist
  - Create desktop windows to visualize the annotations
  - About the data source for the annotations:
    - By default, annotations come from model predictions file
    - If `-file` argument is used, parse a given file as cvat annotations instead
- `merge_cvat_annotations.py`: When importing annotations from cvat, a lot of information is either lost or overwritten such as category_id, info, categories, etc, so this script solves this problem by inputting 2 annotation files and correctly merging the information
  - Input two relative filenames, which are expected to be:
    - File with the correct metadata (auto generated previously by `test.py`)
    - File with the correct labels (which just got manually optimized and imported on cvat)
  - Write new merged file to the current file path (overwrites existing file)
- `main.py`: Main project file which does the following
  - Generate dataset and annotations if they don't exist
  - Train the pretrained model (TODO)
  - Test the application in real-time using the current video device (TODO)

## Practical overview (How to use those scripts)

- First of all download some video files and move them to `data-videos` folder.
- Run `test.py` to build dataset, build dataset annotations and generate initial predictions, as well as visualize those predictions.
- Create a task on [cvat](https://app.cvat.ai) and import the annotations to visualize the predicted bounding boxes and manually optimize them as much as needed.
- Download the updated annotations as "COCO 1.0" and run `merge_cvat_annotations.py` passing the original annotations and the downloaded annotations as arguments.
- Use `train.py` to train the pretrained model.
- Run `python test.py -model yolov8s_trained` to test the trained model against the labels it was trained on.
- Run `main.py` to test the trained model against the current video device.
