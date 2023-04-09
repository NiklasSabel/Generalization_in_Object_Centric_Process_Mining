import warnings
warnings.filterwarnings('ignore')
from src.models.alignment_measure import alignment_measure_states
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle

filename_variant = "../src/data/csv/bpi2017_variant_log.csv"
object_types = ["application","offer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel_variant = ocel_import_factory_csv.apply(file_path=filename_variant, parameters=parameters)

with open("../src/data/csv/bpi_variant_ocpn.pickle", "rb") as file:
    variant_ocpn = pickle.load(file)

print(variant_ocpn)

value = alignment_measure_states(ocel_variant,variant_ocpn)
print(f'Value for the normal ocel_variant log for states {value}')



filename_filtered = "../src/data/filtered_traces/BPI2017-Final_variant_filtered.csv"
object_types = ["application","offer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel_filtered = ocel_import_factory_csv.apply(file_path=filename_filtered, parameters=parameters)

with open("../src/data/csv/bpi_variant_ocpn.pickle", "rb") as file:
    variant_ocpn = pickle.load(file)

print(variant_ocpn)

value = alignment_measure_states(ocel_variant,variant_ocpn)
value

print(f'Value for the filtered ocel_variant log for states {value}')