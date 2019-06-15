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


 
