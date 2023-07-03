from ocpa.algo.conformance.precision_and_fitness import evaluator as quality_measure_factory
import numpy as np
import pickle
import logging
import pandas as pd


def VAE_generalization(ocel_gen, ocpn):
    """
    Calculates the generalization of the generated object-centric event log from the VAE
    :param ocel_gen: generated object-centric event log
    :param ocpn: original object-centric process net
    :return: generalization
    """
    # use precision and fitness from the ocpa package
    precision, fitness = quality_measure_factory.apply(ocel_gen, ocpn)
    print("Precision of IM-discovered net: ",np.round(precision,4))
    print("Fitness of IM-discovered net: ", np.round(fitness,4))
    # calculate generalization as harmonic mean of precision and fitness
    generalization = 2 * ((fitness * precision) / (fitness + precision))
    print("VAE Generalization=", np.round(generalization,4))

    return generalization


logging.basicConfig(filename='VAE_sample_variants.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("*** VAE DS3 Original OCPN**")
ocel_gen = pd.read_pickle('/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS3_ocel_gen.pickle')
logging.info("*** OCEL loaded**")

with open("/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS3_variant_ocpn.pickle", "rb") as file:
    ocpn = pickle.load(file)

logging.info("*** OCPN loaded**")


generalization = VAE_generalization(ocel_gen, ocpn)
print(generalization)

logging.info(f'The value of generalization for VAE original for ds3 ocpn is {generalization}')



logging.info("*** VAE DS3 Original Variant**")

with open("/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS3_variant_ocpn.pickle", "rb") as file:
    variant_ocpn = pickle.load(file)

logging.info("*** OCPN loaded**")

for transition in variant_ocpn.transitions:
    split_string = transition.name.split("_")
    transition.name = split_string[0]

generalization = VAE_generalization(ocel_gen, variant_ocpn)
print(generalization)

logging.info(f'The value of generalization for VAE original for ds3 variant is {generalization}')
