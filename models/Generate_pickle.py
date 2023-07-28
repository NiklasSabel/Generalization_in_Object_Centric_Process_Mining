import warnings
warnings.filterwarnings('ignore')
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import pickle



print('Load file')


filenames = ["order_process"]
sample_sizes = [500,800]
objects = [["order","item","delivery"]]
counter = 0
for filename in filenames:
    object_types = objects[counter]
    counter += 1
    for sample_size in sample_sizes:
        parameters = {"obj_names": object_types,
                      "val_names": [],
                      "act_name": "event_activity",
                      "time_name": "event_timestamp",
                      "sep": ","}
        print(f'Start {sample_size}')
        ocel_variant = ocel_import_factory_csv.apply(file_path=f"../src/data/runtime/variant_logs/{filename}_{sample_size}_variant_log.csv", parameters=parameters)
        print('OCEL variant finished')
        with open(f"../src/data/runtime/{filename}_{sample_size}_ocel_variant.pickle", 'wb') as fp:
            pickle.dump(ocel_variant, fp)
        print(f'Finished {sample_size}')
