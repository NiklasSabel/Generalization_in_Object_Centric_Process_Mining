import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle



filename_filtered = "../src/data/csv/DS3_variant_log.csv"
object_types = ["incident","customer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}

print('Load file')
ocel_filtered = ocel_import_factory_csv.apply(file_path=filename_filtered, parameters=parameters)

print('Save file')

with open("../src/data/csv/DS3_variant.pickle", 'wb') as fp:
	pickle.dump(ocel_filtered, fp)
