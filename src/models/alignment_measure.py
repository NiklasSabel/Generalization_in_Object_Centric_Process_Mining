import numpy as np
from tqdm import tqdm

def filter_case_variants(ocel, variant_column, id_column, save_path):
    """
    Function to filter an object-centric event log such that it only contains one case per case variant of the original log.
    :param ocel: object-centric event log that should be filtered, type: ocel-log
    :param variant_column: column name of the variant column in the log, type: string
    :param id_column: column name of the id column in the log, type: string
    :param save_path: path where the filtered log should be saved, type: string
    """
    # generate the variants of the log by executing the variants method from the ocel class once
    ocel.variants
    # we filter down the dataframe to only keep the first appearance of each variant definition in the variant column
    case_ids = ocel.log.log.drop_duplicates(subset=[variant_column], keep='first')[id_column]
    # we generate a list to keep only the cases (process executions) that contain one of the ids that we filtered before
    # such that we only have one case per variant
    filtered_case_list = [s for s in ocel.process_executions if any(v in s for v in case_ids)]
    # we create a boolean mask indicating which rows contain case ids in the filtered list of sets
    mask = ocel.log.log[id_column].isin([i for s in filtered_case_list for i in s])
    # we apply the mask to the DataFrame to generate the filtered one
    filtered_df = ocel.log.log[mask]
    # in a next step we save the filtered dataframe as a csv file to be able to import it as an ocel-log
    filtered_df.to_csv(save_path, index=False)

def alignment_measure_events(ocel,ocpn):
    """
    Function to calculate the alignment measure based on the used places summing over the events inside an object-centric petri-net.
        :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
        :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
        :return: final value of the formula, type: float rounded to 4 digits
    """
    #list for values in sum of formula
    pnew = []
    # get a list of all activities that have been performed in the log
    log = ocel.log.log.event_activity
    # We only calculate the values for "non-silent" transitions
    transitions = [x for x in ocpn.transitions if not x.silent]
    # dictionary to store each activity as key and a list of its prior states/places as value
    targets = {}
    for arc in tqdm(ocpn.arcs, desc="Check the arcs"):
        # for each arc, check if our target is a valid (non-silent) transition
        if arc.target in transitions:
            # load all the prior places of a valid transition into a dictionary, where the key is the transition and the value
            # a list of all directly prior places
            if arc.target.name in targets:
                targets[arc.target.name].append(arc.source.name)
            else:
                targets[arc.target.name] = [arc.source.name]
    # for each valid transition/event -> for computing reasons(efficiency), we work with a small difference to above
    for event in tqdm(transitions, desc="Save the transitions"):
        # create an empty list where we can store all enabled transitions in the specific prior place
        enabled= []
        # print(event) #used for debugging
        # get the list of all events that are simultaneously in the current state
        for key in targets:
            # we check if the value is the same or if the value of another key is a subset, because then it is also enabled
            if (targets[event.name] == targets[key]) or (set(targets[key]).issubset(set(targets[event.name]))):
                enabled.append(key)
        # number of activities that can be triggered
        w = len(enabled)
        # number of times this state is visited in the log
        n = len(log[log.isin(enabled)])
        # frequency of event that is currently watched
        freq = len(log[log == event.name])
        #print(w) #used for debugging
        #print(n) #used for debugging
        #print(freq) #used for debugging
        # derive the value for the sum in the formula given
        if n >= w+2 :
            pnew.append(freq*(w*(w+1))/(n*(n-1)))
        else:
            pnew.append(freq*1)
    #derive the final generalization value
    return np.round((1 - np.sum(pnew)/len(log)),4)


def alignment_measure_states(ocel,ocpn):
    """
    Function to calculate the alignment measure based on the used places summing over the states inside an object-centric petri-net.
        :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
        :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
        :return: final value of the formula, type: float rounded to 4 digits
    """
    #list for values in sum of formula
    pnew = []
    # We only calculate the values for "non-silent" transitions
    transitions = [x for x in ocpn.transitions if not x.silent]
    # dictionary to store each activity as key and a list of its prior states/places as value
    targets = {}
    # get a list of all activities that have been performed in the log
    log = ocel.log.log.event_activity
    for arc in tqdm(ocpn.arcs, desc="Check the arcs"):
        # for each arc, check if our target is a valid (non-silent) transition
        if arc.target in transitions:
            # load all the prior places of a valid transition into a dictionary, where the key is the transition and the value
            # a list of all directly prior places
            if arc.target.name in targets:
                targets[arc.target.name].append(arc.source.name)
            else:
                targets[arc.target.name] = [arc.source.name]
    #get the list of all possible states for our model
    states = list(targets.values())
    #get the list of all possible keys in our targets dictionary
    keys_list = list(targets.keys())
    #define a counting variable for the number of states
    i = 0
    # for each valid transition/event -> for computing reasons(efficiency), we work with a small difference to above
    for state in tqdm(states, desc="Save the states"):
        # create an empty list where we can store all enabled transitions in the specific prior state
        enabled= []
        #define an empty string that should hold the name of the event we are currently investigating
        event_name = keys_list[i]
        # get the list of all events that are simultaneously in the current state
        for key in targets:
            # we check if the value is the same as the state or if the value of another key is a subset, because then it is also enabled
            if (state == targets[key]) or (set(targets[key]).issubset(set(state))):
                enabled.append(key)

        # number of activities that happened in the state
        w = len(enabled)
        # number of times this state has visited in the log
        n = len(log[log.isin(enabled)])
        # number of times this state was visited in the log
        freq = len(log[log == event_name])
        #print(w) #used for debugging
        #print(n) #used for debugging
        #print(freq) #used for debugging
        if n >= w+2 :
            pnew.append(freq*(w*(w+1))/(n*(n-1)))
        else:
            pnew.append(freq*1)
        #increase the counting variable
        i += 1
        #derive the final generalization value
    return np.round((1 - np.sum(pnew)/len(states)),4)