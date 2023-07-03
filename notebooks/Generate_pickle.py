import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from src.utils import get_happy_path_log, create_flower_model, generate_variant_model



print('Load file')

object_types = ["incident","customer"]

parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel_gen = ocel_import_factory_csv.apply(file_path='../src/data/VAE_generated/DS3_process_sampled.csv', parameters=parameters)


print('Save file GEN DS3')

with open("../src/data/csv/DS3_ocel_gen.pickle", 'wb') as fp:
	pickle.dump(ocel_gen, fp)


print('Finished')
