# A General Approach to Classify Raw Accelerometer Signals for Pump Cavitation Detection

A general approach will be briefly explained here for the classification and identification of the cavitation problem within the hydraulic pumps using signals collected through accelerometers placed inside the pumps.
In essence, the approach that will be described can be applied to any set of temporal data because it does not make use of hand-built features but it is designed specifically for its autonomous extraction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Installation for Windows

```
Python Version 3.6
```
[Anaconda](https://www.anaconda.com/)

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
In the path *path_prefix* you need the *data-raw/DS1* and *data-raw/DS2* folders which inside there are the binary files.


## Running the filter.ipynb
```
.
├── dir1
│   ├── file11.ext
│   └── file12.ext
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── file_in_root.ext
└── README.md
```

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
