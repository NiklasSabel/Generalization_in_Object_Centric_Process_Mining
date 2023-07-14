import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from src.utils import get_happy_path_log, create_flower_model, generate_variant_model




print('Load file')

filenames = ["DS4"]
sample_sizes = [500, 800]
for filename in filenames:
    for sample_size in sample_sizes:
        ocel = ocel_import_factory.apply(
            f"../src/data/runtime/{filename}_{sample_size}.jsonocel")
        with open(f"../src/data/runtime/{filename}_{sample_size}.pickle", 'wb') as fp:
            pickle.dump(ocel, fp)
        ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})
        with open(f"../src/data/runtime/{filename}_{sample_size}_ocpn.pickle", 'wb') as fp:
            pickle.dump(ocpn, fp)
        print(f'Finished {sample_size}')



