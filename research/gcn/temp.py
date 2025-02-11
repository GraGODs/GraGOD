# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.gcn.model import GCN, GCN_PLModule
from models.predict import main
from gragod.types import cast_dataset, cast_model

ckpt_path = "/home/fbello/GraGOD/output/gcn/version_1/best.ckpt"

model_pl = GCN_PLModule.load_from_checkpoint(ckpt_path)
model = model_pl.model

# %%
dataset = cast_dataset("swat")
model = cast_model("gcn")
predict_output = main(
    model, dataset, ckpt_path=ckpt_path, params_file="models/gcn/params.yaml"
)
# %%
