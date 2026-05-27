# %%
from datasets import load_mihaela_service_training_data, load_telco_training_data
from gragod import start_research  # noqa

# %%
# Load the training data
mihaela_data = load_mihaela_service_training_data(
    "Clash_of_Clans", base_path="datasets_files/mihaela"
)
print(mihaela_data)
# %%
telco_data = load_telco_training_data(base_path="datasets_files/telco")
print(telco_data)
# %%
