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

- Input file: train.json

### Method 1: Remove column with `na`

- Output file: data\_processed\_angle\_removed.csv

Convert .json format to .csv format. Returns data frame with 1604 rows, 11253 columns. For each row, the order of the columns are: 'id', 'band\_1' * 5625, 'band\_2' * 5625, 'is\_iceberg'.

Removed a feature `inc_angle` because there were 133 missing data out of 1604 entris. Other columns had no missing values.

```
python data_cleaning.py data/train.json
```

### Method 2: Remove rows with `na` in the column `inc_angle`

- Output file: data\_processed\_rows\_eliminated.csv

```
python data_cleaning2.py data/train.json
```

### Method 3: Add two bands together and Remove rows with `na` in the column `inc_angle`

- Output file: data\_processed\_bands\_combined.csv

```
python data_cleaning3.py data/train.json
```

## Experiment 1 (Without angle column)

### Source Codes

- main\_knn.py
- main\_nn.py
- main\_svm.py
- main\_rf.py
- classification\_algo.py

### Running the codes

```
python main_svm.py -i data/data_processed_angle_removed.csv -kfold 5 -kernel poly
python main_svm.py -i data/data_processed_angle_removed.csv -kfold 5 -kernel linear
python main_svm.py -i data/data_processed_angle_removed.csv -kfold 5 -kernel rbf
python main_svm.py -i data/data_processed_angle_removed.csv -kfold 5 -kernel sigmoid
python main_knn.py -i data/data_processed_angle_removed.csv -kfold 5
python main_rf.py -i data/data_processed_angle_removed.csv -kfold 5
python main_nn.py -i data/data_processed_angle_removed.csv -kfold 5 -activation identity -solver adam
python main_nn.py -i data/data_processed_angle_removed.csv -kfold 5 -activation logistic -solver adam
```

### Results
```
0.753,kernel=poly,C=0.125,gamma=0.0001220703125,degree=1,SVM
0.721,kernel=linear,C=0.125,SVM
0.724,kernel=rbf,C=32.0,gamma=3.0517578125e-05,SVM
0.531,kernel=sigmoid,C=0.125,gamma=3.0517578125e-05,SVM
0.751,n=5,weights=uniform,kNN
0.767,criterion=entropy,n=1024,minss=2,RandomForest
0.636,hls=(32, 64, 32),alpha=0.25,activation=identity,solver=adam,NeuralNetwork
0.531,hls=(16, 16, 16),alpha=0.00390625,activation=logistic,solver=adam,NeuralNetwork
```

## Experiment 2 (Removed rows with 'na' in teh angle)

### Running the codes

```
python main_svm.py -i data/data_processed_rows_eliminated.csv -kfold 5 -kernel poly
python main_svm.py -i data/data_processed_rows_eliminated.csv -kfold 5 -kernel linear
python main_svm.py -i data/data_processed_rows_eliminated.csv -kfold 5 -kernel rbf
python main_svm.py -i data/data_processed_rows_eliminated.csv -kfold 5 -kernel sigmoid
python main_knn.py -i data/data_processed_rows_eliminated.csv -kfold 5
python main_rf.py -i data/data_processed_rows_eliminated.csv -kfold 5
python main_nn.py -i data/data_processed_rows_eliminated.csv -kfold 5 -activation identity -solver adam
```

### Results
```
0.770,kernel=poly,C=0.5,gamma=3.0517578125e-05,degree=1,SVM
0.740,kernel=linear,C=0.125,SVM
0.728,kernel=rbf,C=2.0,gamma=3.0517578125e-05,SVM
0.512,kernel=sigmoid,C=0.125,gamma=3.0517578125e-05,SVM
0.761,n=3,weights=uniform,kNN
0.767,criterion=gini,n=512,minss=4,RandomForest
0.634,hls=(32, 64, 32),alpha=0.5,activation=identity,solver=adam,NeuralNetwork
```

## Experiment 3 (Combine bands together)

### Running the codes

```
python main_svm.py -i data/data_processed_bands_combined.csv -kfold 5 -kernel poly
python main_svm.py -i data/data_processed_bands_combined.csv -kfold 5 -kernel linear
python main_svm.py -i data/data_processed_bands_combined.csv -kfold 5 -kernel rbf
python main_svm.py -i data/data_processed_bands_combined.csv -kfold 5 -kernel sigmoid
python main_knn.py -i data/data_processed_bands_combined.csv -kfold 5
python main_rf.py -i data/data_processed_bands_combined.csv -kfold 5
python main_nn.py -i data/data_processed_bands_combined.csv -kfold 5 -activation identity -solver adam
```

### Results
```
0.724,kernel=poly,C=0.125,gamma=0.0001220703125,degree=1,SVM
0.673,kernel=linear,C=0.125,SVM
0.815,kernel=rbf,C=2.0,gamma=3.0517578125e-05,SVM
0.708,n=3,weights=uniform,kNN
0.753,criterion=gini,n=512,minss=8,RandomForest
0.633,hls=(64, 32, 64),alpha=1.0,activation=identity,solver=adam,NeuralNetwork
```

rbf: experiment with smaller alpha

## Dimensionality Reduction

Reduce dimension of the dataset

### PCA

Reduce the dimension of the bands using PCA.

#### number of dimensions: 100

### Results
```
0.835,kernel=poly,C=0.125,gamma=0.0001220703125,degree=2,SVM
0.862,kernel=rbf,C=8.0,gamma=7.62939453125e-06,SVM
0.773,n=3,weights=uniform,kNN
0.833,criterion=entropy,n=256,minss=8,RandomForest
0.762,hls=(32, 32, 32),alpha=0.0625,activation=identity,solver=adam,NeuralNetwork
```

#### number of dimensions: 50

### Results
```
0.852,kernel=poly,C=0.125,gamma=0.0001220703125,degree=2,SVM
0.861,kernel=rbf,C=512.0,gamma=1.9073486328125e-06,SVM
0.850,criterion=gini,n=1024,minss=2,RandomForest
0.804,n=3,weights=uniform,kNN
0.770,hls=(64, 16, 64),alpha=0.015625,activation=identity,solver=adam,NeuralNetwork
0.763,hls=(16, 16, 16),alpha=0.0078125,activation=tanh,solver=adam,NeuralNetwork
```

## Testing the model on the test dataset

I created the `testdata.csv` file in the Karst server due to the scale of the dataset.

### Reduce dimension of the test data

1. Fit pca on the training set
2. Transform the training set
3. Transform the test set using above fitted(?) pca model

```
python pca.py -i data/data_processed_rows_eliminated.csv -t data/testdata.csv -o data_processed_pca -n 100
```

Training the model using the train set and test them, outputs a `output.csv` with two columns: (id, probability of iceberg)
```
python svm_test.py -i data/data_processed_pca100.csv -t data/test_data_processed_pca100.csv -C 8 -gamma 0.00000762939453125 -kernel rbf
```

- Current ranking: 1270th 
- Log loss: 0.3147



