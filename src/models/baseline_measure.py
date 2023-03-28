import numpy as np


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