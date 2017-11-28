# Statoil/C-CORE Iceberg Classifier Challenge - Ship or Iceberg?

## Team members

Hankyu Jang, Sunwoo Kim, Tony Lam

## Data Description (sourced from [Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data))

In this competition, you will predict whether an image contains a ship or an iceberg. The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

## Data fields

### `train.json`, `test.json`

The data (`train.json`, `test.json`) is presented in json format. The files consist of a list of images, and for each image, you can find the following fields:

- `id` - the id of the image
- `band_1`, `band_2` - the flattened image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are float numbers with unit being dB. Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). More background on the satellite imagery can be found here.
- `inc_angle` - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- `is_iceberg` - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

### `sample_submission.csv`

The submission file in the correct format:

- `id` - the id of the image
- `is_iceberg` - your predicted probability that this image is iceberg.

## Prerequisite

Since the codes are written in Python 3, if you run the code on the campus servers, turn Python 3 module on by running:

```
module load python/3.6.0
```

## Preprocessing

`data_cleaning.py`

- Input file: train.json
- Output file: data\_processed.csv

Convert .json format to .csv format. Returns data frame with 1604 rows, 11253 columns. For each row, the order of the columns are: 'id', 'band\_1' * 5625, 'band\_2' * 5625, 'is\_iceberg'.

Removed a feature `inc_angle` because there were 133 missing data out of 1604 entris. Other columns had no missing values.

```
python data_cleaning.py data/train.json
```

## Experiment 1 (Using Original Data)

### Source Codes

- main\_knn.py
- main\_nn.py
- main\_svm.py
- main\_rf.py
- classification\_algo.py

### Running the codes

```
python main_svm.py -i data/data_processed.csv -kfold 5 -kernel poly
python main_svm.py -i data/data_processed.csv -kfold 5 -kernel linear
python main_svm.py -i data/data_processed.csv -kfold 5 -kernel rbf
python main_svm.py -i data/data_processed.csv -kfold 5 -kernel sigmoid
python main_knn.py -i data/data_processed.csv -kfold 5
python main_nn.py -i data/data_processed.csv -kfold 5 -activation identity -solver adam
python main_nn.py -i data/data_processed.csv -kfold 5 -activation logistic -solver adam
```
