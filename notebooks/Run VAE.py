import warnings
warnings.filterwarnings('ignore')

from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from src.utils import get_happy_path_log, create_flower_model, generate_variant_model, sample_traces, process_log
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
from models.VAE_measure import get_text_data, decode_sequence, create_lstm_vae, VAE_generalization, create_VAE_input
from ocpa.algo.util.filtering.log import case_filtering
from tqdm import tqdm
import numpy as np
import pickle
import logging

logging.basicConfig(filename='VAE_sample_variants.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("*** VAE DS3 Variant**")

filename = "../src/data/jsonocel/DS3.jsonocel"
ocel = ocel_import_factory.apply(filename)
logging.info("*** OCEL loaded**")
ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})
logging.info("*** OCPN OCPA loaded**")


object_types = ["incident","customer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel_gen = ocel_import_factory_csv.apply(file_path='../src/data/VAE_generated/DS3_process_sampled.csv', parameters=parameters)
print("Files Loaded")
logging.info("*** OCEL GEN loaded**")

#generalization = VAE_generalization(ocel_gen, ocpn)
#print(f"Generalization OCPA: {generalization}")
#logging.info(f'The value of generalization for VAE for ds3 ocpa is {generalization}')


filename = "../src/data/jsonocel/DS3.jsonocel"
ots = ["incident","customer"]
flower_ocpn = create_flower_model(filename,ots)
generalization = VAE_generalization(ocel_gen, flower_ocpn)
print(f"Generalization Flower: {generalization}")
logging.info(f'The value of generalization for VAE for ds3 flower is {generalization}')



with open("../src/data/csv/DS3_variant_ocpn.pickle", "rb") as file:
    variant_ocpn = pickle.load(file)

for transition in variant_ocpn.transitions:
    split_string = transition.name.split("_")
    transition.name = split_string[0]

generalization = VAE_generalization(ocel_gen, variant_ocpn)
print(f"Generalization Variant: {generalization}")
logging.info(f'The value of generalization for VAE for ds3 variant is {generalization}')

