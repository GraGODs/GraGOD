# %%

import os
import sys

import pandas as pd

path_to_dataset = "../../datasets_files/swat"
df_train = pd.read_csv(os.path.join(path_to_dataset, "SWaT_Dataset_Normal_v1.csv"))
df_val = pd.read_csv(os.path.join(path_to_dataset, "SWaT_Dataset_Attack_v0.csv"))

# %%
df_train.head()
# %%
df_val.head()
# %%
(df_train["Normal/Attack"] == "Normal").value_counts()
# %%
(df_val["Normal/Attack"] == "Attack").value_counts()


# %%
def load_swat_dataset():
    return df_train, df_val
