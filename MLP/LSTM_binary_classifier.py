import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import CSVLogger

import matplotlib.pyplot as plt
import seaborn as sn

from numpy.random import seed

seed(42)  # keras seed fixing
plt.style.use('ggplot')

path_prefix = "D:/prova/"  # "~ Documents/"  "C:/Users/filip/OneDrive/Documenti/Python/ML-Project/"  #
w = 15  # window used for filtering [5 or 15]
# For the temporal normalization
norm_w = '75k'  # window used for the normalization [25k, 75k or 150k]

# feature columns
sequence_cols = ['P1_x', 'P1_y', 'P1_z', 'P2_x', 'P2_y', 'P2_z']
# class columns
sequence_cols_y = ['category_OK', 'category_STANDING']
# network parameters
sequence_length = 25000
skip_step = 25000
batch_size = 32
STATE = False

"""
Load Data
If OK, IN and/or STAND are not None load the csv files specified on them, split to have train and validation dataset.
"""
def load_data(path, OK=None, IN=None, STAND=None, train_perc=0.75):
    dfs = list()

    # load OK dataset
    temp = list()
    if OK is not None:
        for name in OK:
            temp.append(pd.read_csv(path_prefix + path + name + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    temp = list()
    if IN is not None:
        for name in IN:
            temp.append(pd.read_csv(path_prefix + path + name + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    temp = list()
    if STAND is not None:
        for name in STAND:
            temp.append(pd.read_csv(path_prefix + path + name + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    ret = list()
    ret_val = list()
    # Data preprocessing
    for i, df_split in enumerate(dfs):
        dim = int(len(df_split) * train_perc)
        splitted = np.split(df_split, [dim])
        if len(splitted[0]) > 0:
            ret.append(splitted[0])
        if len(splitted[1]) > 0:
            ret_val.append(splitted[1])

    df = None
    df_val = None
    for i in range(len(ret)):
        if df is not None:
            df = df.append(ret[i], ignore_index=True)
        else:
            df = ret[i]

    for i in range(len(ret_val)):
        if df_val is not None:
            df_val = df_val.append(ret_val[i], ignore_index=True)
        else:
            df_val = ret_val[i]

    return df, df_val


"""
normalization on the entire dataframe
"""
def normalization(dataframe):
    dataframe['cycle_norm'] = dataframe['cycle']
    cols_normalize = dataframe.columns.difference(['time', 'cycle', 'Class'])
    min_max_scaler = MinMaxScaler((-1, 1))

    norm_df = pd.DataFrame(min_max_scaler.fit_transform(dataframe[cols_normalize]),
                           columns=cols_normalize,
                           index=dataframe.index)

    join_df = dataframe[dataframe.columns.difference(cols_normalize)].join(norm_df)

    # One hot encoding of the Class column
    join_df['Class'] = pd.Categorical(join_df['Class'])
    dfDummies = pd.get_dummies(join_df['Class'], prefix='category')
    # join_df = join_df[join_df.columns.difference('Class')].join(dfDummies)
    join_df = pd.concat([join_df, dfDummies], axis=1)
    return join_df.reindex(columns=join_df.columns)


""" 1 - NORMALIZATION ON THE ENTIRE DATASET(switch also the test section in the main)
# Loading filtered data and make the normalization on the entire Dataset
path = "data-raw/DS1/FILTERED/"
test_path = "data-raw/DS2/FILTERED/"
OK_csv = ["OK3_FILTERED_w{}", "OK4_FILTERED_w{}"]
STAND_csv = ["STANDING1_FILTERED_w{}"]
df, df_val = load_data(path, [ok.format(w) for ok in OK_csv], None, [stand.format(w) for stand in STAND_csv])

df = normalization(df)
df_val = normalization(df_val)

OK_test_csv = ["OK_FILTERED_w{}"]
STAND_test_csv = ["STANDING1_FILTERED_w{}", "STANDING2_FILTERED_w{}", "STANDING3_FILTERED_w{}", "STANDING4_FILTERED_w{}"]

"""
# 2 - TEMPORAL NORMALIZATION
# Or load Data preprocessed(window normalized before)
path = "data-raw/DS1/FILTERED/norm/"
test_path = "data-raw/DS2/FILTERED/norm/"
OK_csv = ["OK3_4_w{}_norm_{}"]
STAND_csv = ["STANDING1_w{}_norm_{}"]
df, df_val = load_data(path, [ok.format(w, norm_w) for ok in OK_csv], None, [stand.format(w, norm_w) for stand in STAND_csv])

OK_test_csv = ["OK_w{}_norm_{}"]
STAND_test_csv = ["STANDING1_w{}_norm_{}", "STANDING2_w{}_norm_{}", "STANDING3_w{}_norm_{}", "STANDING4_w{}_norm_{}"]
# -------------------------------------------------------------------------

print("Training shape {}".format(len(df)))
print("Class==OK: {}, {}%".format(len(df[df['Class'] == 'OK']), len(df[df['Class'] == 'OK']) / len(df)))
print("Class==STANDING: {}, {}%".format(len(df[df['Class'] == 'STANDING']),
                                        len(df[df['Class'] == 'STANDING']) / len(df)))
print(df.groupby(['Class'], sort=False).size())

print("Validation shape {}".format(len(df_val)))
print("Class==IN: {}, {}%".format(len(df_val[df_val['Class'] == 'OK']),
                                  len(df_val[df_val['Class'] == 'OK']) / len(df_val)))
print("Class==STANDING: {}, {}%".format(len(df_val[df_val['Class'] == 'STANDING']),
                                        len(df_val[df_val['Class'] == 'STANDING']) / len(df_val)))
print(df_val.groupby(['Class'], sort=False).size())


# Modelling
# Standard Keras model save and load from the current directory
def save_model(mdl, model_filename='model'):
    # serialize model to JSON
    model_json = mdl.to_json()
    with open(model_filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    mdl.save_weights(model_filename + ".h5")
    print("Saved model to disk")


def load_model(model_filename='model'):
    # load json and create model
    json_file = open(model_filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_filename + ".h5")
    print("Loaded model from disk")
    return loaded_model


# Print, if present, the loss and accuracy graph based on the training.log produced at the end of the training phase
def show_graph(file='training.log'):
    history = pd.read_csv(file, sep=',', engine='python')
    dic = {'loss': ['loss', 'val_loss'], 'mse': ['mse', 'val_mse'], 'acc': ['acc', 'val_acc'],
           'categorical_accuracy': ['categorical_accuracy', 'val_categorical_accuracy']}
    # summarize history for accuracy
    plt.xlabel('epoch')
    for k in dic.keys():
        if k in history.keys():
            leg = []
            for val in dic[k]:
                leg.append(val)
                if val in history.keys():
                    plt.plot(history[val])

                plt.title('model {}'.format(k))

            plt.legend(leg, loc='upper left')
            plt.ylabel(k)

            plt.show()


# Used from predict_model function to get the real label of signals examples
def getYTrue(input_df, seq_length, cols_y, skip_step):
    ret_y_true = []
    data_array_y = input_df[cols_y].values
    to = int(np.floor((((len(input_df) - seq_length) / skip_step) + 1)))
    for i in range(0, to):
        f = i * skip_step
        ret_y_true.append(data_array_y[f + seq_length - 1])

    return np.asarray(ret_y_true)


#  The standard Keras ResetModelStates
class ResetModelStates(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, dataframe, seq_length, seq_cols, seq_cols_y, batch_size, skip_step,
                 shuffle=False, title='TITLE'):
        """Initialization"""
        self.data = dataframe
        self.seq_length = seq_length
        self.seq_cols = seq_cols
        self.seq_cols_y = seq_cols_y
        self.batch_size = batch_size
        self.skip_step = skip_step
        self.shuffle = shuffle
        self.number_batch_4_epoch = int(
            (np.floor((((len(self.data) - self.seq_length) / self.skip_step) + 1))) / self.batch_size)
        self.indexes = np.arange(self.number_batch_4_epoch * self.batch_size)

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.number_batch_4_epoch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        data_array_x = self.data[self.seq_cols].values
        data_array_y = self.data[self.seq_cols_y].values
        f = idx * self.skip_step * self.batch_size

        px = list()
        py = list()
        for i in range(self.batch_size):
            if self.shuffle:
                iddx = self.indexes[idx * self.batch_size + i]
                f = (iddx * self.skip_step)
                px.append(data_array_x[f:f + self.seq_length])
                py.append(data_array_y[f])
            else:
                px.append(data_array_x[f + (i * self.skip_step):f + (i * self.skip_step) + self.seq_length])
                py.append(data_array_y[f + (i * self.skip_step)])

        x = np.array(px)
        y = np.array(py)
        return x, y


# Creation of the LSTM model
def create_model(b_s, state=False):
    model = Sequential()

    model.add(LSTM(
        input_shape=(sequence_length, len(sequence_cols)),
        batch_size=b_s,
        units=100,
        return_sequences=False,
        stateful=state))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(units=len(sequence_cols_y), activation='sigmoid'))
    return model


def load_model_hdf5(file):
    model = create_model(batch_size)
    model.load_weights(file+".hdf5")
    return model


# Creation of the model and execute the fit_generator
def model_generation(sequence_length, batch_size, tdg, vdg, ne, state=False, optimizer=None):
    rms = ResetModelStates()
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    Path(path_prefix+'checkpoints').mkdir(parents=True, exist_ok=True)
    mck = ModelCheckpoint(filepath=path_prefix+'/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                          monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    if optimizer is None:
        optimizer = Adam(lr=0.0005)

    model = create_model(batch_size, state)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    model.fit_generator(generator=tdg, validation_data=vdg, epochs=ne, verbose=1, callbacks=[csv_logger, rms, mck, es], shuffle=False)

    save_model(model)
    show_graph()
    return model


def plot_confusion_matrix(cm, targets, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(targets))
    plt.xticks(tick_marks, targets, rotation=45)
    plt.yticks(tick_marks, targets)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def run_experiment():
    optimizer = Adam(lr=0.0005)
    num_epochs = 35

    model = model_generation(sequence_length, batch_size, train_data_generator, validate_data_generator,
                             num_epochs, STATE, optimizer)

    return model


def print_confusion_matrix(y_true, y_pred, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=['OK', 'STAND'],
               yticklabels=['OK', 'STAND'])  # font size
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluate_model(model, gen):
    scores = model.evaluate(gen, verbose=1)
    print('Accurracy: {}'.format(scores[1]))


def predict_model(model, gen, title):
    # make predictions and compute confusion matrix
    y_pred_prob = model.predict(gen, verbose=1)
    y_true = getYTrue(gen.data, sequence_length, sequence_cols_y, skip_step)
    # print(y_true)
    # print(y_pred_prob)

    y_true_dopo = y_true.argmax(axis=1)
    y_pred_dopo = y_pred_prob.argmax(axis=1)
    y_true_dopo = y_true_dopo[:len(y_pred_dopo)]

    print_confusion_matrix(y_true_dopo, y_pred_dopo, title)

    # compute precision and recall
    precision = precision_score(y_true_dopo, y_pred_dopo, average='micro')
    recall = recall_score(y_true_dopo, y_pred_dopo, average='micro')
    print('precision = ', precision, '\n', 'recall = ', recall)


train_data_generator = DataGenerator(df, sequence_length, sequence_cols, sequence_cols_y, batch_size=batch_size,
                                     skip_step=skip_step, shuffle=True)
validate_data_generator = DataGenerator(df_val, sequence_length, sequence_cols, sequence_cols_y, batch_size=batch_size,
                                        skip_step=skip_step, shuffle=True)

if __name__ == '__main__':

    # 1 - Load model file
    # model = load_model("model")
    # OR 2 Load from hdf5
    # model = load_model_hdf5(path_prefix+"checkpoints/weights.14-0.05")
    """ necessary if a model is loaded
    optimizer = Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    show_graph()
    """
    # OR 3 - create and train a new model (comment if you download a model)
    model = run_experiment()
    # ----

    validate_data_generator = DataGenerator(df_val, sequence_length, sequence_cols, sequence_cols_y,
                                            batch_size=batch_size, skip_step=skip_step, shuffle=False)
    evaluate_model(model, validate_data_generator)
    predict_model(model, validate_data_generator, "Validation CM")
    del df, df_val

    # Test
    # 1 - filtered files and online normalization(switch also the normalization section at the beginning)
    """
    df_test, _ = load_data("data-raw/DS2/FILTERED/", [ok.format(w) for ok in OK_test_csv],
                           None,
                           [stand.format(w) for stand in STAND_test_csv], 1)
    df_test = normalization(df_test)
    """
    # 2 - OR load the preprocessed normalized data - NO online normalization
    df_test, _ = load_data("data-raw/DS2/FILTERED/norm/", [ok.format(w, norm_w) for ok in OK_test_csv],
                           None,
                           [stand.format(w, norm_w) for stand in STAND_test_csv], 1)

    test_gen = DataGenerator(df_test, sequence_length, sequence_cols, sequence_cols_y,
                             batch_size=32, skip_step=skip_step, shuffle=False)
    # --- --- ---

    predict_model(model, test_gen, "Test - CM")
