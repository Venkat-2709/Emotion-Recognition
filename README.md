# Emotion-Recognition

## Classifying pain from Images.

### Step 1: Installing the Requirements. 

```
pip install -r requirements.txt 
```

### Step 2: Project Structure.
Setup the project folder according to this structure. 
```
    |--data_folder
    |     |--Training
    |          |--No pain (Images)
    |          |--Pain (Images)
    |     |--Testing
    |          |--No pain (Images)
    |          |--Pain (Images)
    |     |--Validation
    |          |--No pain (Images)
    |          |--Pain (Images)
    |--main.py
    |--classifier.py
    |--Output_folder
    |    |--128
    |        |--Training
    |            |--No pain (Images)
    |            |--Pain (Images)
    |        |--Testing
    |            |--No pain (Images)
    |            |--Pain (Images)
    |        |--Validation
    |            |--No pain (Images)
    |            |--Pain (Images)
    |    
    |    |--64
    |        |--Training
    |            |--No pain (Images)
    |            |--Pain (Images)
    |        |--Testing
    |            |--No pain (Images)
    |            |--Pain (Images)
    |        |--Validation
    |            |--No pain (Images)
    |            |--Pain (Images)
```

### Step 2: Execute
```
python main.py WIDTH HEIGHT <Data Folder>
```

WIDHT - Width of image you want to crop.
HEIGHT - Height of image you want to crop. 

After cropping the images with OpenCV Casscasde Classifier only the faces of images will be stored in the Output_folder.

The cropped images will be of size WIDTH x HEIGHT.

###### Note: This might take few minutes to finish executing (20-30 mins) Since the data very large to process.

After preprocessing the images are then trained using Transfer Learning VGG19 model. 

### Output
```
For 128 X 128 size images -
    Found 9788 images belonging to 2 classes.
    Found 2003 images belonging to 2 classes.
    Found 3545 images belonging to 2 classes.
        Classification for images of size 128 X 128 -
            The Accuracy of the model: 0.6290563941001892
            The Precision of the model: 0.4065359477124183
            The Recall of the model: 0.3820638820638821
            The F1 Score of the model: 0.39392020265991134
            The Confusion Matrix: [[735 454], [503 311]]
```

### License

[MIT](https://github.com/Venkat-2709/Emotion-Recognition/blob/master/LICENSE)
