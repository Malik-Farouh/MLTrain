# TrkPID

## Introduction

TrkPID is a machine learning algorithm is trained to differentiate between conversion electrons and cosmic ray muons using information from both the tracker and the calorimeter

## Workflow

The python code provided in the file [TrackPIDTrain.py](TrkPIDTrain.py) is used to:
* make a dataset containing conversion electrons and cosmic ray muons
* define the neural network structure
* train the algorithm and save the model weights into an ONNX file named "TrackPID.onnx"
* test the algorithm, providing performance metrics
* generate plots to provide more information on the dataset, how the training went, and how the model perform

Once the model is trained and the weights are saved in an ONNX file, this file can be used by TMVA:SOFIE to generate the inference code that can be used in Offline (for details about this process, check [this documentation](https://github.com/Mu2e/MLTrain/blob/main/TrkQual/README.md#converting-a-model-for-use-in-offline)).

## Version history

This version has been trained using MDC2020au datasets, and tested on MDC2020au and MDC2020aw, generated using Offline v11_00_00 and EventNtuple v06_07_00.
