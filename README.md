# A General Approach to Classify Raw Accelerometer Signals for Pump Cavitation Detection

A general approach will be briefly explained here for the classification and identification of the cavitation problem within the hydraulic pumps using signals collected through accelerometers placed inside the pumps.
In essence, the approach that will be described can be applied to any set of temporal data because it does not make use of hand-built features but it is designed specifically for its autonomous extraction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Installation for Windows

```
Python Version 3.6
Anaconda 1.9.12
```

Create the following directory structure.\
Dataset Directory Tree Structure
```
.
~ Documents
├── checkpoints
└── data-raw 
    ├── DS1
    |   ├── 20130110082835continuo.dat
    |   ├── 20130110082907continuo.dat
    |   ├── 20130110083352continuo.dat
    |   ├── 20130110084454continuo.dat
    |   ├── 20130110084754continuo.dat
    |   ├── 20130110085127continuo.dat
    |   ├── 20130110090319continuo.dat
    |   └── FILTERED
    |       └── norm
    └── DS2
        ├── 20130111092657continuo.dat
        ├── 20130111092838continuo.dat
        ├── 20130111093220continuo.dat
        ├── 20130111094402continuo.dat
        ├── 20130111095117continuo.dat
        ├── 20130111095449continuo.dat
        └── FILTERED
            └── norm
```

### Installing

Create a new conda enviroment
```
conda create -n ML_SACMI python=3.6
```

Install all the packeges listed in requirements.txt
```
conda install --yes --file requirements.txt
```

## Running the from_raw_to_csv_and_some_Esa.ipynb

To use the models proposed it is necessary to convert the binary to csv files.\
Set *path_prefix='~ Documents'*(the data-raw folder) and run all the cells\
Dataset Directory Tree Structure updated:
```
.
~ Documents
├── checkpoints
└── data-raw 
    ├── DS1
    |   ├── 20130110082835continuo.dat
    |   ├── 20130110082907continuo.dat
    |   ├── 20130110083352continuo.dat
    |   ├── 20130110084454continuo.dat
    |   ├── 20130110084754continuo.dat
    |   ├── 20130110085127continuo.dat
    |   ├── 20130110090319continuo.dat
    |   ├── OK1.csv
    |   ├── OK2.csv
    |   ├── OK3.csv
    |   ├── OK4.csv
    |   ├── IN1.csv
    |   ├── STANDING1.csv
    |   ├── STANDING2.csv
    |   └── FILTERED
    |       └── norm
    └── DS2
        ├── 20130111092657continuo.dat
        ├── 20130111092838continuo.dat
        ├── 20130111093220continuo.dat
        ├── 20130111094402continuo.dat
        ├── 20130111095117continuo.dat
        ├── 20130111095449continuo.dat
        ├── OK1.csv
        ├── IN1.csv
        ├── STANDING1.csv
        ├── STANDING2.csv
        ├── STANDING3.csv
        ├── STANDING4.csv
        └── FILTERED
            └── norm
```


## Running the filter.ipynb
Set *path_prefix='~ Documents'*(the data-raw folder) and run all the cells to filter the csv files\
Dataset Directory Tree Structure updated:
```
.
~ Documents
├── checkpoints
└── data-raw 
    ├── DS1
    |   ├── 20130110082835continuo.dat
    |   ├── 20130110082907continuo.dat
    |   ├── 20130110083352continuo.dat
    |   ├── 20130110084454continuo.dat
    |   ├── 20130110084754continuo.dat
    |   ├── 20130110085127continuo.dat
    |   ├── 20130110090319continuo.dat
    |   ├── OK1.csv
    |   ├── OK2.csv
    |   ├── OK3.csv
    |   ├── OK4.csv
    |   ├── IN1.csv
    |   ├── STANDING1.csv
    |   ├── STANDING2.csv
    |   └── FILTERED
    |       ├── OK1_FILTERED_w5.csv & OK1_FILTERED_w15.csv
    |       ├── OK2_FILTERED_w5.csv & OK2_FILTERED_w15.csv
    |       ├── OK3_FILTERED_w5.csv & OK3_FILTERED_w15.csv
    |       ├── OK4_FILTERED_w5.csv & OK4_FILTERED_w15.csv
    |       ├── IN1_FILTERED_w5.csv & IN1_FILTERED_w15.csv
    |       ├── STANDING1_FILTERED_w5.csv & STANDING1_FILTERED_w15.csv
    |       ├── STANDING2_FILTERED_w5.csv & STANDING2_FILTERED_w15.csv
    |       └── norm
    └── DS2
        ├── 20130111092657continuo.dat
        ├── 20130111092838continuo.dat
        ├── 20130111093220continuo.dat
        ├── 20130111094402continuo.dat
        ├── 20130111095117continuo.dat
        ├── 20130111095449continuo.dat
        ├── OK1.csv
        ├── IN1.csv
        ├── STANDING1.csv
        ├── STANDING2.csv
        ├── STANDING3.csv
        ├── STANDING4.csv
        └── FILTERED
            ├── OK_FILTERED_w5.csv & OK_FILTERED_w15.csv
            ├── IN_FILTERED_w5.csv & IN_FILTERED_w15.csv
            ├── STANDING1_FILTERED_w5.csv & STANDING1_FILTERED_w15.csv
            ├── STANDING2_FILTERED_w5.csv & STANDING2_FILTERED_w15.csv
            ├── STANDING3_FILTERED_w5.csv & STANDING3_FILTERED_w15.csv
            ├── STANDING4_FILTERED_w5.csv & STANDING4_FILTERED_w15.csv
            └── norm
```

## Running the normalization.py
Two kind of normalization will be compared(on the enire dataset vs the window normalization). The normalization.py is necessary to perform a window normalization on all the files and to store them on the file system. We could decide if applying a typical normalization on all entire dataset(we can do at runtime because it's fast) or using the preprocessed normalized csv files(procedure too slow at runtime).\
Set *path_prefix='~ Documents'*(the data-raw folder) and run the file.(It may require few minutes!)

```
.
~ Documents
├── checkpoints
└── data-raw 
    ├── DS1
    |   ├── 20130110082835continuo.dat
    |   ├── 20130110082907continuo.dat
    |   ├── 20130110083352continuo.dat
    |   ├── 20130110084454continuo.dat
    |   ├── 20130110084754continuo.dat
    |   ├── 20130110085127continuo.dat
    |   ├── 20130110090319continuo.dat
    |   ├── OK1.csv
    |   ├── OK2.csv
    |   ├── OK3.csv
    |   ├── OK4.csv
    |   ├── IN1.csv
    |   ├── STANDING1.csv
    |   ├── STANDING2.csv
    |   └── FILTERED
    |       ├── OK1_FILTERED_w5.csv & OK1_FILTERED_w15.csv
    |       ├── OK2_FILTERED_w5.csv & OK2_FILTERED_w15.csv
    |       ├── OK3_FILTERED_w5.csv & OK3_FILTERED_w15.csv
    |       ├── OK4_FILTERED_w5.csv & OK4_FILTERED_w15.csv
    |       ├── IN1_FILTERED_w5.csv & IN1_FILTERED_w15.csv
    |       ├── STANDING1_FILTERED_w5.csv & STANDING1_FILTERED_w15.csv
    |       ├── STANDING2_FILTERED_w5.csv & STANDING2_FILTERED_w15.csv
    |       └── norm
    |           ├── OK1_2_w5_norm_25k.csv & OK1_2_w15_norm_25k.csv
    |           ├── OK1_2_w5_norm_75k.csv & OK1_2_w15_norm_75k.csv
    |           ├── OK1_2_w5_norm_150k.csv & OK1_2_w15_norm_150k.csv
    |           ├── OK3_4_w5_norm_25k.csv & OK3_4_w15_norm_25k.csv
    |           ├── OK3_4_w5_norm_75k.csv & OK3_4_w15_norm_75k.csv
    |           ├── OK3_4_w5_norm_150k.csv & OK3_4_w15_norm_150k.csv
    |           ├── IN1_w5_norm_25k.csv & IN1_w15_norm_25k.csv
    |           ├── IN1_w5_norm_75k.csv & IN1_w15_norm_75k.csv
    |           ├── IN1_w5_norm_150k.csv & IN1_w15_norm_150k.csv
    |           ├── STANDING1_w5_norm_25k.csv & STANDING1_w15_norm_25k.csv
    |           ├── STANDING1_w5_norm_75k.csv & STANDING1_w15_norm_75k.csv
    |           ├── STANDING1_w5_norm_150k.csv & STANDING1_w15_norm_150k.csv
    |           ├── STANDING2_w5_norm_25k.csv & STANDING2_w15_norm_25k.csv
    |           ├── STANDING2_w5_norm_75k.csv & STANDING2_w15_norm_75k.csv
    |           └── STANDING2_w5_norm_150k.csv & STANDING2_w15_norm_150k.csv
    └── DS2
        ├── 20130111092657continuo.dat
        ├── 20130111092838continuo.dat
        ├── 20130111093220continuo.dat
        ├── 20130111094402continuo.dat
        ├── 20130111095117continuo.dat
        ├── 20130111095449continuo.dat
        ├── OK1.csv
        ├── IN1.csv
        ├── STANDING1.csv
        ├── STANDING2.csv
        ├── STANDING3.csv
        ├── STANDING4.csv
        └── FILTERED
            ├── OK_FILTERED_w5.csv & OK_FILTERED_w15.csv
            ├── IN_FILTERED_w5.csv & IN_FILTERED_w15.csv
            ├── STANDING1_FILTERED_w5.csv & STANDING1_FILTERED_w15.csv
            ├── STANDING2_FILTERED_w5.csv & STANDING2_FILTERED_w15.csv
            ├── STANDING3_FILTERED_w5.csv & STANDING3_FILTERED_w15.csv
            ├── STANDING4_FILTERED_w5.csv & STANDING4_FILTERED_w15.csv
            └── norm
                ├── OK_w5_norm_25k.csv & OK_w15_norm_25k.csv
                ├── OK_w5_norm_75k.csv & OK_w15_norm_75k.csv
                ├── OK_w5_norm_150k.csv & OK_w15_norm_150k.csv
                ├── IN_w5_norm_25k.csv & IN_w15_norm_25k.csv
                ├── IN_w5_norm_75k.csv & IN_w15_norm_75k.csv
                ├── IN_w5_norm_150k.csv & IN_w15_norm_150k.csv
                ├── STANDING1_w5_norm_25k.csv & STANDING1_w15_norm_25k.csv
                ├── STANDING1_w5_norm_75k.csv & STANDING1_w15_norm_75k.csv
                ├── STANDING1_w5_norm_150k.csv & STANDING1_w15_norm_150k.csv
                ├── STANDING2_w5_norm_25k.csv & STANDING2_w15_norm_25k.csv
                ├── STANDING2_w5_norm_75k.csv & STANDING2_w15_norm_75k.csv
                ├── STANDING2_w5_norm_150k.csv & STANDING2_w15_norm_150k.csv
                ├── STANDING3_w5_norm_25k.csv & STANDING3_w15_norm_25k.csv
                ├── STANDING3_w5_norm_75k.csv & STANDING3_w15_norm_75k.csv
                ├── STANDING3_w5_norm_150k.csv & STANDING3_w15_norm_150k.csv
                ├── STANDING4_w5_norm_25k.csv & STANDING4_w15_norm_25k.csv
                ├── STANDING4_w5_norm_75k.csv & STANDING4_w15_norm_75k.csv
                └── STANDING4_w5_norm_150k.csv & STANDING4_w15_norm_150k.csv
```

### Run the 1° Model - LSTM_binary_clissifier.py
Set *path_prefix='~ Documents'*(the data-raw folder).\
Set the filtering window *w=15 or 5*.\
Set the normalization window *norm_w=25k, 75k or 150k*(Only if we enable the temporal normalization - 2).

We have to decide between:
* 1 - NORMALIZATION ON THE ENTIRE DATASET 
* 2 - TEMPORAL NORMALIZATION
Comment/Uncomment the related sections at the beginning(for loading) and insiede the main(for testing).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
