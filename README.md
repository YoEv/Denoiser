# Denoiser (2024)

## Introduction
This is a time-domian denoiser project based on modified Conv-TasNet architecture. 

## References  
This project get the reference from Demucs project and Conv-TasNet Architechture. 
Two papers sourses:
[Real Time Speech Enhancement in the Waveform Domain](https://arxiv.org/abs/2006.12847)
[Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)

The proposed model is based on an encoder-separator-decoder architecture with temporal convolutional blocks. It is optimized on time domains, using l1 loss functions.

Empirical evidence shows that it is capable of removing various kinds of real-life background noise including human speech background noise, supermarket backgound noise with music and sounds, real-life room reverb, instructional noise from outside.

After the training process, I enhanced the noisy audio samples from the inferencing stage through ConvTasNet with the best weights that get trained. 

Audio samples can be found here: 

## Architecture 

### Traning:

```
Start Training (Solver.train)
        ↓
Initialize Model, Optimizer, and Training Config
        ↓
For Each Epoch (Up to args.epochs):
    ↓
    Set Model to Training Mode (self.model.train)
    ↓
    For Each Batch in Training Data:
        ↓
        1. Load Noisy and Clean Audio
        2. Apply Data Augmentation (if enabled)
        3. Pass Noisy Audio through Model to Get Estimate
        4. Calculate Loss:
            - Primary Loss 
            - Optional: STFT Loss (if args.stft_loss is enabled)
        5. Backpropagate Gradients (loss.backward)
        6. Update Model Weights (optimizer.step)
        ↓
    Calculate Training Loss for the Epoch
    ↓
    (Optional) Perform Cross-Validation:
        - Pass Validation Data through Model
        - Calculate Validation Loss
    ↓
    Save Best Model and Checkpoint (if args.checkpoint is enabled)
    ↓
    (Optional) Evaluate on Test Set and Enhance Samples (if eval_every)
        - Evaluate Metrics (e.g., PESQ, STOI)
        - Enhance and Save Samples (enhance)
        ↓
Log Training and Validation Metrics
        ↓
End Epoch
        ↓
Save Training History to File
        ↓
End Training

```

### Inference:
```
Raw Audio Waveform (Batch, 1, Time)
        ↓
    Encoder (1D Conv)
        ↓
 Feature Maps (Batch, N, Time)
        ↓
Separation Module (TemporalConvNet)
 - Stacked TemporalBlocks (B=16)
 - Repeated TemporalConvNets (R=8)
        ↓
 Feature Maps (Batch, N, Time)
        ↓
    Decoder (1D Transpose Conv)
        ↓
Separated Audio Output (Batch, 1, Time)

```


### Configuration

We use [Hydra][hydra] to control all the training configurations. If you are not familiar with Hydra
we recommend visiting the Hydra [website][hydra-web].

Generally, Hydra is an open-source framework that simplifies the development of research applications
by providing the ability to create a hierarchical configuration dynamically.

The config file with all relevant arguments for training our model can be found under the `conf` folder.
Notice, under the `conf` folder, the `dset` folder contains the configuration files for the different datasets. You should see a file named `debug.yaml` with the relevant configuration for the debug sample set.


### Checkpointing
Small Conv-TasNet
Large Conv-TasNet

### Setting up a new dataset

You need to generate the relevant `.json`files in the `egs/`folder.
For that purpose you can use the `python -m denoiser.audio` command that will
scan the given folders and output the required metadata as json.
For instance, if your noisy files are located in `$noisy` and the clean files in `$clean`, you can do

```bash
out=egs/mydataset/tr
mkdir -p $out
python -m denoiser.audio $noisy > $out/noisy.json
python -m denoiser.audio $clean > $out/clean.json
```

## Usage

### 1. Data Structure
The data loader reads both clean and noisy json files named: `clean.json` and `noisy.json`. These files should contain all the paths to the wav files to be used to optimize and test the model along with their size (in frames).
You can use `python -m denoiser.audio FOLDER_WITH_WAV1 [FOLDER_WITH_WAV2 ...] > OUTPUT.json` to generate those files.
You should generate the above files for both training and test sets (and validation set if provided). Once this is done, you should create a yaml (similarly to `conf/dset/debug.yaml`) with the dataset folders' updated paths.
Please check [conf/dset/debug.yaml](conf/dset/debug.yaml) for more details.


### 2. Training
Training is simply done by launching the `train.py` script:

```
./train.py
```

This scripts read all the configurations from the `conf/config.yaml` file.


#### Logs

Logs are stored by default in the `outputs` folder. Look for the matching experiment name.
In the experiment folder you will find the `best.th` serialized model, the training checkpoint `checkpoint.th`,
and well as the log with the metrics `trainer.log`. All metrics are also extracted to the `history.json`
file for easier parsing. Enhancements samples are stored in the `samples` folder (if `noisy_dir` or `noisy_json`
is set in the dataset).

### 3. Evaluating

Evaluating the models can be done by:

```
python -m denoiser.evaluate --model_path=<path to the model> --data_dir=<path to folder containing noisy.json and clean.json>
```
Note that the path given to `--model_path` should be obtained from one of the `best.th` file, not `checkpoint.th`.

### 4. Denoising

Generating the enhanced files can be done by:

```
python -m denoiser.enhance --model_path=<path to the model> --noisy_dir=<path to the dir with the noisy files> --out_dir=<path to store enhanced files>
```
Notice, you can either provide `noisy_dir` or `noisy_json` for the test data.
Note that the path given to `--model_path` should be obtained from one of the `best.th` file, not `checkpoint.th`.

