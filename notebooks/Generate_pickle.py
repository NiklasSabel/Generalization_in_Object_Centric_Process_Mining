import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle

filename_filtered = "../src/data/filtered_traces/DS4_variant_filtered.csv"
object_types = ["Payment application","Control summary","Entitlement application","Geo parcel document","Inspection","Reference alignment"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}

print('Load file')
ocel_filtered = ocel_import_factory_csv.apply(file_path=filename_filtered, parameters=parameters)

print('Save file')

with open("../src/data/csv/DS4_variant_filtered.pickle", 'wb') as fp:
	pickle.dump(ocel_filtered, fp)
