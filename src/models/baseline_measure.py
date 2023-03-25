import numpy as np


def baseline_measure(ocel,ocpn,activity_name,activity_id):
    """
       Function to calculate the baseline measure based on the used places inside an object-centric petri-net.
       :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
       :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
       :param activity_name: column name of the activity column in the log, type: string
       :param activity_id: column name of the activity id in the log, type: string
       :return: flower model, type: object-centric petri net
    """
    # We define an empty dictionary where we can store the places that have a valid transition as target
    execution_dict = {}
    # We define an empty list, where we store the values used in the baseline formula
    executions = []
    # We only calculate the values for "non-silent" transitions
    transitions = [x for x in ocpn.transitions if not x.silent]
    # We get all the places from the object-centric petri net given
    places = ocpn.places
    # We store the counts how often each activity was executed in the log
    transitions_counts = ocel.log.log.groupby(activity_name).count()[activity_id]
    # for each arc in the our petri net, we check if the target is a valid transition and then store source:target as
    # key:value-pair in our dictionary
    for arc in ocpn.arcs:
        if arc.target in transitions:
            execution_dict[arc.source] = arc.target
    for place in places:
        # we check if each place in our places has a valid target transition (exlcuding silent transitions as target)
        if place in execution_dict:
            # if the target is a valid transition, we calculate the value from the baseline formula for this transition and store it
            executions.append(1/np.sqrt(transitions_counts.loc[transitions_counts.index == execution_dict[place].name][0]))
    #we give back the final valid formula
    return np.round(1-np.sum(executions)/len(places),4)