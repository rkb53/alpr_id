# Character Recognition - TensorFlow-Slim

Framework for training and inferencing models to be used for Indonesian License Plate detection. This section is for Character Recognition.

## Getting Started

This program was developed on Ubuntu 16.04. Before running this in your system, be sure to setup the environment as per the requirements below.

### Dependencies

* Python v3.5.2
* [Tensorflow-GPU](https://www.tensorflow.org/install/install_linux) - Tensorflow, machine learning framework from Google. We use the version with GPU compatibility. It is reccomended to use the docker method of installing Tensorflow.
* [OpenCV v3.3.0] - Framework for image processing
* [imgaug] (https://github.com/aleju/imgaug) - Framework for image augmentation


### Environment

* nvidia-docker - As instructed in tensorflow's website
* [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim) - This is the basic framework that we are using

## Running the Program

This program consists of three process, augmenting dataset, training the model and using the resulting model for inferencing.

### Augmentation 

We need to augment (and replicate) the dataset as an alternative to cope up low number of dataset so we will have big enough identical data. The parameter If you already had big number of dataset of license plate characters (ex: 200000 files), you do not have to replicate the dataset, but you still need to make sure that the datasets are balanced in each class, hence you may set the `--multiple` parameter near the number of the class that has the maximum files. Then, you can set the `--limit` parameter as a threshold to trim all files in each class to specific amount.

```shell
$ DATA_DIR=/tmp/data/dataset_alpr
$ python3 augmentation.py \
    --input="${DATA_DIR}" \
	--output="${DATA_DIR + "_augmented/alpr"}" \
    --multiple=18000 \
    --limit=8000	
```
When the script finishes you will find a new folder with `"_augmented"` at the end of the file name. This folder consists of `"alpr"` folder as an input to the next step of training.

### Training

The instruction is basically the same as [TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim), but with few adjustment.

#### Converting to TFRecord format

Before converting dataset to TFRecord format, you need to set parameters in `datasets/alpr.py`. There is a variable `SPLIT_TO_SIZES` and you need to adjust the proportion of train and validation data. Ten percent of train data to be converted into validation data is advised.

```shell
$ DATA_DIR=/tmp/data/dataset_alpr_augmented/
$ python3 download_and_convert_data.py \
    --dataset_name=alpr \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
alpr_train-00000-of-00036.tfrecord
...
alpr_train-00035-of-00036.tfrecord
alpr_validation-00000-of-00036.tfrecord
...
alpr_validation-00035-of-00036.tfrecord
labels.txt
```
These represent the training and validation data, sharded over 36 files each. You will also find the `$DATA_DIR/labels.txt` file which contains the mapping from integer labels to class names.

#### Training a model from scratch
Below are the instructions for training

```shell
DATASET_DIR=/tmp/data/dataset_alpr_augmented/
TRAIN_DIR=/tmp/train_logs
python3 /workspace/tensorflow/models/research/slim/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=alpr \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR}  \
    --model_name=vgg_16 \
    --batch_size=128 \
    --learning_rate=0.01
```
This process may take several days, depending on your hardware setup.
Note that by using different GPU, you need to pay attention at the maximum `--batch_size` the GPU can take.

 |GPU|Nets|Maximum batch size|
 |:---|:---|---:|
 |Tesla V100|VGG 16|128|
 |GTX 1080 Ti|VGG 16|64|

#### Evaluating performance of a model

To evaluate the performance of a model, you can use the `eval_image_classifier.py` script, as shown below.

```shell
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/model.ckpt- # Example
python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=alpr \
    --dataset_split_name=validation \
    --model_name=vgg_16
```

### Inferencing


## Authors and Contributors

This documentation was made by:
* Christoporus Deo Putratama,
  github: [cputratama](https://github.com/cputratama)


Project Authors include:

* Muhammad Aldo Aditiya Nugroho
* Christoporus Deo Putratama
* Dhimas Bintang Kusumawardhana
* Yosua Sepri Andasto
* Yudi Pratama

## Acknowledgments


