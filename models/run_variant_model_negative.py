import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle
import logging
import numpy as np
from tqdm import tqdm
import os
import re


def baseline_measure(ocel,ocpn,activity_name,activity_id):
    """
       Function to calculate the baseline measure based on the used places inside an object-centric petri-net.
       :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
       :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
       :param activity_name: column name of the activity column in the log, type: string
       :param activity_id: column name of the activity id in the log, type: string
       :return: final value of the formula, type: float rounded to 4 digits
    """
    # We define an empty list, where we store the values used in the baseline formula
    executions = []
    # We only calculate the values for "non-silent" transitions
    transitions = [x for x in ocpn.transitions if not x.silent]
    # We store the counts how often each activity was executed in the log
    transitions_counts = ocel.log.log.groupby(activity_name).count()[activity_id]
    # for each transition in the log, we calculate the value for the baseline formula
    for transition in transitions_counts.index:
        executions.append(1/np.sqrt(transitions_counts.loc[transitions_counts.index == transition]))
    #we give back the final valid formula
    return np.round(1-np.sum(executions)/len(transitions),4)



# Set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define a file handler
file_handler = logging.FileHandler('results_variant_model_negative_events.log')
file_handler.setLevel(logging.INFO)

# Define a log formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Log a message
logger.info('This log file shows the results of the variant model baseline for the DS4 log')
with open("/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS4_variant_cel.pickle", "rb") as file:
    ocel_variant = pickle.load(file)

with open("/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS4_variant_ocpn.pickle", "rb") as file:
    variant_ocpn = pickle.load(file)

value = baseline_measure(ocel_variant,variant_ocpn,'event_activity','event_id')
print(value)
logger.info("*** Evaluate ***")
logger.info('The value of generalization for DS3 is %s', value)