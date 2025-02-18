<div align="center">

# 🚀 GraGOD

<img width="300" alt="GraGOD Logo" src="https://github.com/user-attachments/assets/80aafe71-3db6-4aac-9829-fd95f773caf1" />

### 🔍 **Anomaly Detection with Graph Neural Networks (GNNs)**  
*A modern approach to time-series anomaly detection using GNN techniques.*

---

</div>

GraGOD is a modern approach to time-series anomaly detection using GNN techniques.
It is a PyTorch-based framework that provides a flexible and modular architecture for building and training GNN models for anomaly detection.
The objective of this project is to facilitate the development and testing of new models and datasets for anomaly detection, providing a standard way to organize the code and the data.

All this project is developed by [Federico Bello](https://github.com/federicobello) and [Gonzalo Chiarlone](https://github.com/gonzalochiarlone). 
It's part of our thesis to get our Engineering Degree at [La Facultad de Ingenieria](https://www.fing.edu.uy/) de la [Universidad de la República](https://www.uru.edu.ar/).


## 📦 Installation

The project requirements are managed with [poetry](https://python-poetry.org/). After installing poetry, install the project dependencies with:

```bash
poetry install
```

Then, to activate the project environment, run:

```bash
poetry shell
```


## 📦 Usage

### Running the Scripts

#### 📦 Training

The training script is `models/train.py`. It can be run with the following command:

```bash
python models/train.py --model <model_name> --dataset <dataset_name> --params_file <params_file>
```

The `params_file` is an optional argument. If not provided, the script will load the file `models/<model_name>/params.yaml`.

The output of the training script is a folder under `<log_dir>/<model_name>/version_<version>`, where `<version>` is an automatically generated version number.
All the other parameters are loaded from the `params_file`.


#### 📦 Predicting

The predicting script is `models/predict.py`. It can be run with the following command:

```bash
python models/predict.py --model <model_name> --dataset <dataset_name> --params_file <params_file>
```

The `params_file` is an optional argument. If not provided, the script will load the file `models/<model_name>/params.yaml`.

The model is loaded from the path `<ckpt_folder>/best.ckpt`, which is specified in the `params_file`.


#### 📦 Tuning

The tuning script is `models/tune.py`. It can be run with the following command:

```bash
python models/tune.py --model <model_name> --dataset <dataset_name> --params_file <params_file>
```

The `params_file` is an optional argument. If not provided, the script will load the file `models/<model_name>/params.yaml`.

### 📦 CLI

The project also provides a CLI to visualize the metrics and the predictions. To see the available commands, run:

```bash
gragod --help
```

## Supported Models

The project currently supports the following models:

- `gcn`: A graph convolutional network based on TAGConv
- `gdn`: Graph Deviation Network, implementation based on [this repository](https://arxiv.org/pdf/2106.06947) and [this paper](https://arxiv.org/pdf/2106.06947)
- `mtad_gat`: Multi-Task Anomaly Detection with GAT, implementation based on [this repository](https://github.com/ML4ITS/mtad-gat-pytorch) and [this paper](https://arxiv.org/pdf/2009.02040)
- `gru`: A GRU-based model

## Supported Datasets

The project currently supports the following datasets:

- `swat`: Standard Water Treatment dataset, found [here](https://github.com/yzhao062/anomaly-detection-resources)
- `telco`: Telecommunications dataset, found [here](https://iie.fing.edu.uy/investigacion/grupos/anomalias/)

## Metrics

The project supports the calculation of metrics for each time series or for the whole system. Different metrics are calculated taking this into account.

All the metric calculations are done in the `gragod/metrics/calculator.py` file.

Current supported metrics are:

- `precision`: Precision metric
- `recall`: Recall metric
- `f1_score`: F1 score metric
- `range_based_precision`: Range-based precision metric
- `range_based_recall`: Range-based recall metric
- `range_based_f1_score`: Range-based F1 score metric
- `vus_roc`: VUS-ROC metric
- `vus_pr`: VUS-PR metric

The range based metrics are the ones defined in the [PRTS paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf) and the VUS metrics are the ones defined in the [VUS paper](https://github.com/TheDatumOrg/VUS).

## Project Structure

The project is organized as follows:

- `models/`: Contains the implementation of the models and script files to train, predict and tune them.
Each model has its own folder with the implementation of the model and its own `params.yaml` file.
- `datasets/`: Contains the implementation of the datasets and its processing functions.
- `gragod/`: Contains the CLI and the utility functions.
- `datasets_files/`: Contains the files for the datasets.
- `research/`: Contains random and outdated scripts of interesting experiments and visualizations. You probably won't be able to run them.

## Contact

For any questions or feedback, please contact us at `fedebello13@gmail.com` or `gonzalochiarlone@gmail.com`.
