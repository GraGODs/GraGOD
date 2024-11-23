# GraGOD

Anomaly detection with GNNs.

## Running the project

Start virtual enviroment:

`poetry shell`

Run the project with the `gragod` command:

To check that the CLI is working:

`gragod test-cli`

To check the available commands:

`gragod --help`

## Model Training

There are 3 scripts for training the implemented models:

- `models/gcn/train.py`: Script for training the GCN model.
- `models/gdn/train.py`: Script for training the GDN model.
- `models/mtad_gat/train.py`: Script for training the MTAD-GAT model.

To execute these scripts, use the following command format:

```bash
python path/to/script.py --params_file path/to/params
```

All the params are stored in the `params.yaml` file in each model's directory.

Each script will output the trained model in the `output/{model_name}/version_{version_number}` directory.

### Tensorboard

Each script has a Tensorboard built in. To run Tensorboard, use the following command:

```bash
tensorboard --logdir output
```

Here you can see the training and validation losses for each model.

## Model Predictions

There are 3 scripts for generating predictions with the implemented models:

- `models/gcn/predict.py`: Script for generating predictions using the GCN model.
- `models/gdn/predict.py`: Script for generating predictions using the GDN model.
- `models/mtad_gat/predict.py`: Script for generating predictions using the MTAD-GAT model.

To execute these scripts, use the following command format:

```bash
python path/to/script.py --params_file path/to/params --ckpt_path path/to/ckpt
```

Each script will output the predictions in the `output/{model_name}/version_{version_number}` directory that matches the ckpt path.

## Running Tests

To run the tests for the project, use the following command:

`poetry run pytest`

This will execute all the test cases defined in your project. Ensure that you have `pytest` installed in your virtual environment.