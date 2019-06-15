# Face Recognition

This project build with different files. Separate python files are run to achieve different tasks.

1. settings.py -  *Hold all the settings for entire project.* 
2. prepare.py - *Use to separate raw dataset into desirable ratio.*
3. implement.py - *Training model with processed data and saving model in .pt file.*
4. gtk.py - *Starts up gtk application for face recognition.*
5. dataloaders.py - *Contain pytorch data loading from process images.*
6. haar.py - *Displays haar face detection examples.*
7. network.py - *CNN network build using pytorch.*
8. show_batches.py - *Displays batch images from dataloaders*
9. transforms.py - *Pytorch transform to get face detection with haar cascade*


## Installation

Installation after cloning git repository.

```p
mkdir face-recognition
cd face-recognition
git clone git@github.com:roshanshrestha01/Face-Recognition.git app
virtualenv venv
source venv/bin/activate
cd app/
pip install -r requirements.txt
```

## Data Split

Split data into test data and traing data set. Stores images in processed data with pass parameters.

```bash
python prepare.py 6 4
```

Above commands sets 6 images for training in train directory and 4 images for testing in test directory.


## Implement

Running train and validation of model.

```bash 
python implement.py
```


Trains model and output orl_databse_faces.pt when validation loss is decreased. Also gives confusion-matrix.xls.


## GTK application

GTK desktop application has major three button capture image, train and predict video

### Capture Image
Add subject name in input. This name is use as directory name and label to store captured iamges.

Capture image button open up webcam input. Might have to change source
opencv video source in settings.py.

When video frame is open. Press "C" in keyboard with face is detected in video which store image in
directory name set in subject name input box.

### Train model

Click Train model button which uses pretraind model orl_database_faces.pt and added new image from capture
directory. After completion an alert is made.

### Predict Video

Clicking predict video opens up video frame. Whan face is detection prediction is done and label is
written at top right corner of detected face.



![Predit image](image/predit.png#center)




 
