import warnings
warnings.filterwarnings('ignore')

from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
from models.negative_events_measure_without_weighting import negative_events_without_weighting
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

value = negative_events_without_weighting (ocel_variant, variant_ocpn)
print(value)