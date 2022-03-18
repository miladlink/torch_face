# Simple Pytorch Face Recognition

* This is the a `Face Datection & Recognition Package` that make it easy to use.
* Thanks to [facenet-pytorch](https://github.com/timesler/facenet-pytorch) & [OpenCV](https://github.com/opencv/opencv-python) for their powerful libraries which uses at the core of this package.

## Table of contents

* [Table of contents](#table-of-contents)
* [Installation](#installation)
* [Tutorial](#tutorial)
    * [Data](#data)
    * [Image](#image)
    * [Camera](#camera)
    * [Video](#video)
* [Examples](#examples)
* [Refcences](#refrences)

## Installation

```bash
git clone https://github.com/miladlink/torch_face
cd torch_face
pip install -r requirements.txt
```
(this section will be updated)

## Tutorial

The usage of this package is **very simple** and you can detect and recognize each image with few lines of code. some [examples](#examples) were bring to get better application of `torch_face` package

### Data

After installation you just make a directory of `persons` like below tree which was mentioned
**note:** each name should have at least one image

```bash
├── persons
│   ├── Name1
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── Name2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── image3.jpg
│   ├── Name3
│   │   └── image1.jpg
│   ├── Name4
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── ...
│   └── NameN
│       └── imagen.jpg
└── aligned

```

**Making this two directories are only prepration!**

```python
persons_dir = 'data/persons'
faces_dir = 'data/aligned'
```

After creating directories you should build encoding dictionary for each name with just two lines of code and it saves as `.pt ` file to reduce inference time to future use.

```python
from torch_face import faceDetectionRecognition

fdr = faceDetectionRecognition(persons_dir, faces_dir)
encoding_dict = fdr.build_face_storage() # it saved in data.pt file
```

if new name or person wanted to **add** in data collection (`data.pt`) you just add a image of new person in a directory and do the following code:

```python
encode_dir = 'data/data.pt'
fdr = faceDetectionRecognition(persons_dir, faces_dir, encode_dir)
fdr.addFaces('new face', 'Milad')
```

### Image

After building encoding data uou can detect and recognize each image contain of name in `data.pt` very easy!

```python
# Image
img = 'data/multiface.jpg' # or URL, OpenCV, PIL
# Inference
results = fdr.predict(img, encoding_dict)
# Results
results.print() # or .show(), .save(), locations(), crop_faces(save=True)
```

**Visualize**

```python
import cv2
cv2.imshow('MultiFace', results.show())
cv2.waitKey(0)
```

![saved_pred](https://user-images.githubusercontent.com/81680367/158949140-19614559-3fdc-414e-9bde-f45438c8bb17.jpg)

**Plot Cropped Faces**

```python
results.plot()
```

![plot](https://user-images.githubusercontent.com/81680367/158949264-046d7481-2c35-4a87-858a-9ab8c2183653.png)

**landmarks**

Moreover you can add landmarks to faces

```python
results = fdr.predict(img, encoding_dict, landmarks=True)
```

![saved_pred1](https://user-images.githubusercontent.com/81680367/158949164-978bf23e-93b4-4f83-b637-0225b7b913e8.jpg)


### Camera

As you know you can recognize each **online** or **offline** videos just like images. following code uses for camera:

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    results = fdr.predict(img, encoding_dict, landmarks=True)
    cv2.imshow('face detector', results.show())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

![camera](https://user-images.githubusercontent.com/81680367/158981339-6214a3b6-a2e7-48e6-a83d-170455102d5f.gif)


### Video

In camera section if you change video index(0) to video name you can process video faces just like camera. [here](https://github.com/miladlink/torch_face/tree/master/examples/video.py) is complete example of detection and recognition in videos **with saving result**.

## Examples

For all of tutorials there are complete [Examples](https://github.com/miladlink/torch_face/tree/master/examples) that uses [this data](https://github.com/miladlink/torch_face/tree/master/data) and get [this results](https://github.com/miladlink/torch_face/tree/master/results)

## References

https://github.com/timesler/facenet-pytorch