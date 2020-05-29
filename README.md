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

Create the following directory struceture.
Dataset Directory Tree Structure
```
.
~ Documents
├── data-raw
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

To use the models proposed it is necessary to convert the binary to csv files.
Dataset Directory Tree Structure updated:
```
.
~ Documents
├── data-raw
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


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
