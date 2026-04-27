# TrackPIDTrain.py
# Make dataset, train and test artificial neural network for TrackPID
# This code works with MDC2020au and MDC2020aw datasets generated using Offline v11_00_00, EventNtuple v06_07_00.
# Author: Leo Borrel
# Date: 2025-10-29

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import pandas as pd
import tensorflow as tf
import json
import tf2onnx
import onnx


CE_dataset_name = "nts.lborrel.CeMLeadingLogOnSpillTriggered.MDC2020au_perfect_v1_3.root"  # CE MDS2020au dataset used for training
CE_csv_name = "array_test_CE.csv"
MU1_dataset_name = "nts.lborrel.CosmicCRYSignalAllOnSpillTriggered.MDC2020au_perfect_v1_3_part1.root"   # Cosmic MDS2020au dataset used for training (divided in 2 parts because too large)
MU1_csv_name = "array_test_MU_1.csv"
MU2_dataset_name = "nts.lborrel.CosmicCRYSignalAllOnSpillTriggered.MDC2020au_perfect_v1_3_part2.root"   # Cosmic MDS2020au dataset used for training (part 2)
MU2_csv_name = "array_test_MU_2.csv"

# Switches to turn on and off depending on which task needs to be performed
# Import the EventNtuple tree from the ROOT file, apply the cuts, and save the trimmed dataset into a csv file (one for conversion electron, one for cosmic muons)
# Use only once when using a new dataset
switch_make_dataset = False
# Train the machine learning model
# If false, it will use the model saved in a specific file named ""
switch_train = False
# Plot the result and the performance of the trained model
switch_plot = True


def import_evtntuple(filename, list_branches):
    # Import EvenNtuple root file and transform it into an awkward array

    file = uproot.open(filename)
    tree = file["EventNtuple/ntuple"]

    array = tree.arrays(list_branches, library='ak')

    #print("# of events: ", ak.num(array, axis=0))
    #print("# of tracks: ", ak.count(array['trk','trk.status']))

    return array


def apply_cut(array, array_mc, particle):
    # Apply cuts on the dataset by iterating over events, tracks and track segments ; only keep a selected number of branches useful for training and testing
    # array: awkward array containing the reco branches
    # array_mc: awkward array containing the monte carlo branches
    # particle: 'e' for conversion electron dataset, 'mu' for cosmic muon dataset

    if particle == "e":
        mc_pdg = 11
        label_particle = 1
    elif particle == "mu":
        mc_pdg = 13
        label_particle = 0

    data_array = []
    for i_evt in range(ak.num(array, axis=0)):  # iterate over events
        evt_it = array[i_evt]
        for i_trk in range(ak.num(evt_it['trk','trk.status'], axis=0)):   # iterate over tracks
            if evt_it['trk','trk.pdg'][i_trk] != 11:    # mask pdg hypothesis
                continue
            if array_mc['trkmcsim','pdg'][i_evt,i_trk,0] != mc_pdg:     #mask mc pdg
                continue
            if evt_it['trkcalohit','trkcalohit.active'][i_trk] == 0:    # calo hit exists
                continue
            if evt_it['trkqual','trkqual.result'][i_trk] < 0.2:    # mask TrkQual
                continue
            trk_it = evt_it['trksegs'][i_trk]
            for i_trksegs in range(ak.num(trk_it['sid'], axis=0)):  # iterate over track segments
                trksegs_it = trk_it[i_trksegs]
                if trksegs_it['sid'] != 1:  # mask sid
                    continue
                if trksegs_it['mom','fCoordinates','fZ'] < 0:   # mask downstream
                    continue
                if trksegs_it['mom','mag'] < 80 or trksegs_it['mom','mag'] > 130:    # mask momentum
                    continue
                data_array.append({
                    'trkqual': evt_it['trkqual','trkqual.result'][i_trk],
                    'mom': trksegs_it['mom','mag'], 'time': trksegs_it['time'],
                    'edep': evt_it['trkcalohit','trkcalohit.edep'][i_trk],
                    'dt': evt_it['trkcalohit','trkcalohit.dt'][i_trk],
                    'trkdepth': evt_it['trkcalohit','trkcalohit.trkdepth'][i_trk],
                    'pocaX': evt_it['trkcalohit','trkcalohit.poca.fCoordinates.fX'][i_trk],
                    'pocaY': evt_it['trkcalohit','trkcalohit.poca.fCoordinates.fY'][i_trk],
                    'pocaZ': evt_it['trkcalohit','trkcalohit.poca.fCoordinates.fZ'][i_trk],
                    'pocamomX': evt_it['trkcalohit','trkcalohit.mom.fCoordinates.fX'][i_trk],
                    'pocamomY': evt_it['trkcalohit','trkcalohit.mom.fCoordinates.fY'][i_trk],
                    'pocamomZ': evt_it['trkcalohit','trkcalohit.mom.fCoordinates.fZ'][i_trk],
                    'label': label_particle,
                    })

    df_array = pd.DataFrame(data_array)
    print(df_array)
    print(df_array.describe())

    return df_array


def make_dataset(particle, dataset_name, csv_name):
    # Import EventNtuple data and apply a set of cuts ; the cut dataset with only useful branches is saved as a csv file to be used later in the training and test
    # particle: 'e' for conversion electron dataset ; 'mu' for cosmic muon dataset
    # dataset_name: path to the ROOT file containing the EventNtuple tree
    # csv_name: name of the csv file in which the trimmed dataset will be saved in ; this is used to access the data easier later for training

    branches_reco = ['trk','trkqual','trksegs','trksegpars_lh','trkcalohit']
    branches_mc = ['trkmc','trkmcsim','trksegsmc']
    #array_evt = import_evtntuple(dataset_name, ['evtinfo'])
    array = import_evtntuple(dataset_name, branches_reco)
    array_mc = import_evtntuple(dataset_name, branches_mc)

    # make mc time modulo event time
    array_mc['trksegsmc','time_mod'] = np.mod(array_mc['trksegsmc','time'], 1695)
    
    # make momentum magnitude branches
    array['trksegs','mom','mag'] = np.sqrt((array['trksegs','mom','fCoordinates','fX'])**2 + (array['trksegs','mom','fCoordinates','fY'])**2 + (array['trksegs','mom','fCoordinates','fZ'])**2)
    array_mc['trksegsmc','mom','mag'] = np.sqrt((array_mc['trksegsmc','mom','fCoordinates','fX'])**2 + (array_mc['trksegsmc','mom','fCoordinates','fY'])**2 + (array_mc['trksegsmc','mom','fCoordinates','fZ'])**2)

    # make branches for PID
    #array['trkpid.deltaE'] = array['trkcalohit','trkcalohit.edep'] - array['trksegs','mom','mag']
    #array['trkcalohit','trkcalohit.dt'].show()

    df_array = apply_cut(array, array_mc, particle)

    # Save cut array into csv file
    df_array.to_csv(csv_name, index=True)


def train_model(dataframe):
    # Create the multilayer perceptron neural network
    PID_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), batch_size=32),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    # Setup loss, optimizer, metrics, and early stop condition for the model
    model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model_optimizer = tf.keras.optimizers.Adam()
    model_metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.AUC(from_logits=False)]
    PID_model.compile(loss = model_loss, optimizer = model_optimizer, metrics = model_metrics)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", start_from_epoch = 100, patience = 20, restore_best_weights = True, verbose = 1)

    print(PID_model.summary())

    n_epochs = 500
    train_history = PID_model.fit(dataframe[features], dataframe['label'], epochs = n_epochs, validation_split=0.1, callbacks=[early_stop])
    train_history = train_history.history   # extract the training history (loss as function of epochs)
    # Save model and training history
    PID_model.save("PID_model.keras")
    with open("train_history.json",'w') as history_file:
        json.dump(train_history, history_file)

    # export model in onnx format to be able to use it with SOFIE inference code ; manually enter name and shape of input and output for SOFIE
    PID_model.output_names = ['output']
    onnx_signature = [tf.TensorSpec(input.shape, dtype=input.dtype, name=input.name) for input in PID_model.inputs]
    onnx_model, _ = tf2onnx.convert.from_keras(PID_model, input_signature=onnx_signature)
    onnx.save(onnx_model, "TrackPID.onnx")

    return PID_model


def make_results(model, dataset, dataset_name, threshold = 0.5):
    # Print model performances
    results = model.evaluate(dataset[features], dataset['label'])
    print("\n", dataset_name, "loss,", dataset_name, "accuracy,", dataset_name, "AUC:", results, "\n")

    dataset['prediction'] = model.predict(dataset[features])
    dataset['predict_label'] = (dataset['prediction'] > threshold).astype(int)

    # Create confusion matrix
    confusion_matrix = tf.math.confusion_matrix(dataset['label'], dataset['predict_label'], num_classes=2)
    print("\n Confusion matrix: \n [ True negative (correctly labeled cosmic muons) ; False positive (mislabeled cosmic muons) ] \n [ False negative (mislabeled conversion electrons) ; True positive (correctly labeled conversion electron) ]\n", confusion_matrix)

    true_negative , false_positive = confusion_matrix[0].numpy()
    false_negative , true_positive = confusion_matrix[1].numpy()

    TPR = true_positive / (true_positive + false_negative)
    TNR = true_negative / (true_negative + false_positive)

    print("\n", dataset_name, " dataset results:\n")
    print("True Positive Rate (correctly labeled conversion electrons / all conversion electrons): ", 100*TPR, "%")
    print("True Negative Rate (correcly labeled cosmic muons / all cosmic muons): ", 100*TNR, "%")
    print("False Positive Rate (mislabeled conversion electrons / all conversion electrons): ", 100*(1-TPR), "%")
    print("False Negative Rate (mislabeled cosmic muons / all cosmic muons): ", 100*(1-TNR), "%\n")

    return dataset, results, confusion_matrix


## Main code

if switch_make_dataset:
    make_dataset('e', CE_dataset_name, "CE_temp.csv")
    make_dataset('mu', MU_dataset_name, "MU_temp.csv")

df_CE = pd.read_csv(CE_csv_name, index_col=0)
df_MU1 = pd.read_csv(MU1_csv_name, index_col=0)
df_MU2 = pd.read_csv(MU2_csv_name, index_col=0)    # if the cosmic dataset is too big, separate it in 2 pieces and add it it the concatenation list in the next line
df = pd.concat([df_CE, df_MU1, df_MU2], axis=0)

# Make input features

df['deltaE'] = df['edep'] - df['mom']
df['rpoca'] = np.sqrt(np.power(df['pocaX'],2) + np.power(df['pocaY'],2))
df['trkdir'] = ( df['pocaX'] * df['pocamomX'] + df['pocaY'] * df['pocamomY'] ) / ( np.sqrt(np.power(df['pocaX'],2) + np.power(df['pocaY'],2)) * np.sqrt(np.power(df['pocamomX'],2) + np.power(df['pocamomY'],2)) )
features = ['deltaE','rpoca','trkdir','dt']
n_features = len(features)
df_feature = df[features+['label']].copy()


# Shuffle and divide the dataset into a testing dataset with 5000 entries and a training dataset with the rest of the events
df_shuffle = df_feature.sample(frac=1)
df_test = df_shuffle.iloc[:5000,:]
df_train = df_shuffle.iloc[5000:,:]


if switch_train:
    PID_model = train_model(df_train)
else:   # Use an already trained model saved in keras format
    PID_model = tf.keras.models.load_model("PID_model.keras")
    with open("train_history.json",'r') as history_file:    # open file containing the training history to plot later
        train_history = json.load(history_file)
    n_epochs = len(train_history['loss'])

df_train,results_train,confusion_matrix_train = make_results(PID_model, df_train, "train", 0.5)
df_test,results_test,confusion_matrix_test = make_results(PID_model, df_test, "test", 0.5)

# Create datasets based on MC particle information
df_train_e = df_train[df_train["label"] == 1]
df_train_mu = df_train[df_train["label"] == 0]
df_test_e = df_test[df_test["label"] == 1]
df_test_mu = df_test[df_test["label"] == 0]


## Plots

def plot_dataset(csv_name, particle):
    df = pd.read_csv(csv_name, index_col=0)

    # plot of training features
    fig,ax = plt.subplots(1,1)
    ax.hist(df['mom'], bins=100)
    ax.set_xlabel("reco mom [MeV]")
    ax.set_title("Reconstructed momentum of "+particle)

    fig,ax = plt.subplots(1,1)
    ax.hist(df['edep']-df['mom'], bins=100)
    ax.set_xlabel("deltaE [MeV]")
    ax.set_title("Difference between calorimeter cluster energy and track momentum of "+particle)

    fig,ax = plt.subplots(1,1)
    ax.hist(df['dt'], bins=100)
    ax.set_xlabel("deltaT [MeV]")
    ax.set_title("Difference between calorimeter cluster time and tracker time of "+particle)


def plot_feature(dataset_e, dataset_mu, feature, scale = 'linear'):
    # Plot of a branch 
    min_x = min(min(dataset_e[feature]), min(dataset_mu[feature]))
    max_x = max(max(dataset_e[feature]), max(dataset_mu[feature]))

    fig,ax = plt.subplots(1,1)
    ax.hist(dataset_e[feature], color='b', alpha=0.5, range=(min_x,max_x), bins=100, density=False)
    ax.hist(dataset_mu[feature], color='r', alpha=0.5, range=(min_x,max_x), bins=100, density=False)

    ax.set_xlabel(feature)
    ax.set_xlim(min_x-0.05*(max_x-min_x), max_x+0.05*(max_x-min_x))
    ax.set_yscale(scale)
    ax.set_ylabel("# of events")
    ax.set_title(feature)
    ax.legend(["conversion electrons", "cosmic muons"], loc='best')


def plot_ROC(dataset):
    # Plot the ROC (Receiver Operating Characteristic) curve

    n_points = 101
    # Make a list of threshold values not equally distant (follow a power law to have more points close to 0 and 1
    list_threshold = np.concatenate((0.5 * np.power(np.linspace(0,1,n_points//2+1),4), 1 - 0.5 * np.power(np.linspace(1,0,n_points//2+1),4)), axis=0)
    list_threshold = np.delete(list_threshold, n_points//2+1)

    true_negative = np.zeros(n_points)
    false_positive = np.zeros(n_points)
    false_negative = np.zeros(n_points)
    true_positive = np.zeros(n_points)
    dataset['predict_label_temp'] = dataset['label']

    for i in range(n_points):
        dataset['predict_label_temp'] = (dataset['prediction'] >= list_threshold[i]).astype(int)
        confusion_matrix_temp = tf.math.confusion_matrix(dataset['label'], dataset['predict_label_temp'], num_classes=2)

        true_negative[i], false_positive[i] = confusion_matrix_temp[0].numpy()
        false_negative[i], true_positive[i] = confusion_matrix_temp[1].numpy()

    TPR = true_positive / (true_positive + false_negative)
    TNR = true_negative / (true_negative + false_positive)
    FNR = 1 - TPR
    FPR = 1 - TNR

    purity = true_positive / (true_positive + false_positive)
    significance = true_positive / np.sqrt(true_positive + false_positive)
    max_significance_idx = np.nanargmax(significance)

    print("\n Max significance at threshold = ", list_threshold[max_significance_idx])
    print("\n Accuracy at this threshold = ", TPR[max_significance_idx], "\n")

    fig,ax = plt.subplots(1,1)
    ax.plot(TPR, TNR, '-b')
    ax.set_xlabel("Signal efficiency (true positive rate)")
    ax.set_ylabel("Background rejection (true negative rate)")
    ax.set_title("ROC curve")

    fig,ax = plt.subplots(1,1)
    ax.plot(list_threshold, TPR, '-k')
    ax.plot(list_threshold, TNR, '-b')
    ax.plot(list_threshold, purity, '-g')

    ax2 = ax.twinx()
    ax2.plot(list_threshold, significance, '-r')

    ax.set_xlabel("Cut threshold value")
    ax.set_ylabel("Efficiency / Purity")
    ax2.set_ylabel("Significance")
    ax.legend(["Signal efficiency", "Background rejection", "Signal purity"], loc="lower left")
    ax2.legend(["Significance = S/sqrt(S+B)"], loc="lower right")
    
    return dataset


def plot_history(history_file, result):
    # Plot loss history
    with open(history_file, 'r') as json_file:
        history = json.load(json_file)
    fig,ax = plt.subplots(1,1)
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.plot(len(history["loss"]), result[0], marker='o', linestyle='None')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Binary cross entropy loss")
    ax.legend(["Train", "Validation","Test"], loc='best')


# Comment on and off to choose which group of plots to display
if switch_plot:
    plot_dataset(CE_csv_name,"conversion electrons")
    #plot_dataset(MU1_csv_name,"cosmic muons")
    for feature in features:
        plot_feature(df_test_e, df_test_mu, feature)
    plot_feature(df_train_e, df_train_mu, "prediction", 'log')
    plot_feature(df_test_e, df_test_mu, "prediction", 'log')
    df_test = plot_ROC(df_test)
    plot_history("train_history.json", results_test)

    plt.show()

