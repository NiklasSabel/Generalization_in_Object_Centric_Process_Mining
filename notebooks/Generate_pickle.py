import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from src.utils import get_happy_path_log, create_flower_model, generate_variant_model



print('Load file')

filename = "../src/data/jsonocel/BPI2017-Final.jsonocel"
ocel = ocel_import_factory.apply(filename)
ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})

with open("../src/data/csv/BPI_log.pickle", 'wb') as fp:
	pickle.dump(ocel, fp)

with open("../src/data/csv/BPI_ocpn.pickle", 'wb') as fp:
	pickle.dump(ocpn, fp)


happy_path__ocel = get_happy_path_log(filename)

happy_path_ocpn = ocpn_discovery_factory.apply(happy_path__ocel, parameters={"debug": False})

print('Save file happy')


with open("../src/data/csv/bpi_ocpn_happy.pickle", 'wb') as fp:
	pickle.dump(happy_path_ocpn, fp)

print('Save file flower')

ots = ["application","offer"]
flower_ocpn = create_flower_model(filename,ots)

with open("../src/data/csv/bpi_ocpn_flower.pickle", 'wb') as fp:
	pickle.dump(happy_path_ocpn, fp)
