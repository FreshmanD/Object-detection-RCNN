# Object-Detection-RCNN Project

This project using RCNN (Region-CNN) for object detection. 

The dataset is from [CrowdAI](https://github.com/udacity/self-driving-car/tree/master/annotations), it contains 9,421 images with 72,064 entries of labels in three classes (car, truck, and pedestrain). 

### Project Setup
The pre-trained model is ```faster_rcnn_resnet101_coco``` available on TensorFlow website.

**1. Set up the environment**
* Install ```Ubuntu 18.04```
* Install ```Nvidia driver 390``` and ```Nvidia Docker 2```
* Pull the ```tensorflow:1.12.0-gpu-py3``` image and run it using runtime of ```nvidia```
* Get into the container
* Install the TensorFlow Object Detection APIs and COCO APIs

**2. Prepare the scripts needed**
* Modify the [generate_tfrecord.py](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) to fit the needs of the file
* Construct ```label_map.pbtxt``` manually

**3. Prepare ```tfrecord``` file using the dataset**
* Download the dataset mentioned above and extract it using ```tar -xvzf```
* Download the corrected ```.csv``` file
* Split the ```.csv``` file into ```train.csv``` and ```test.csv```
* Used ```generate_tfrecord.py``` to convert the csv files into tfrecord

**4. Retrieve pre-trained model**
* Download the pre-trained model from TensorFlow website
* Extract

**5. Config pipeline**
* Modify the ```pipeline.config``` file in the model
* Set the ```classnames```, ```tfrecords```, and other parameters

**6. Training**
* Set environment parameters
* Run model_main.py with parameters to train
* Launch Tensorboard for monitoring

**7. Inference on specific images**
* Use the helper code from ```object_detection_tutorial.ipynb```
* Set the path to test images and image parameters
* Show image with inferred labels

### Result Evaluation
The model took 60 hours to train. After 50,000 steps of training, the Mean Average Precision (mAP) reached 0.3693. The mAP for large objects reached 0.5546.

<img src="" width="300">