# %%
import temporian as tp

from datasets.telco import load_data, load_df, load_tp, load_training_data
from gragod import start_research  # noqa

TELCO_PATH = "datasets_files/telco"
# %%
X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = load_data(
    TELCO_PATH
)
# %%
print(f"X_train shape: {X_train.shape}")
print(f"X_train_labels shape: {X_train_labels.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_val_labels shape: {X_val_labels.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test_labels shape: {X_test_labels.shape}")
# %%
X_train, X_val, X_test, X_train_labels, X_val_labels, X_test_labels = (
    load_training_data(TELCO_PATH, normalize=True, clean=False)
)
# %%
print(f"X_train shape: {X_train.shape}")
print(f"X_train_labels shape: {X_train_labels.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_val_labels shape: {X_val_labels.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test_labels shape: {X_test_labels.shape}")
# %%
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
    TELCO_PATH
)
# %%
df_train.head()
# %%
es_train, es_label_train, es_val, es_label_val, es_test, es_label_test = load_tp(
    TELCO_PATH
)
# %%

a = es_train["TS1"]
plot = a.simple_moving_average(tp.duration.seconds(10))
plot.plot()

# %%
