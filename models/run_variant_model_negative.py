import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import time
import logging
def filter_silent_transitions(dic,silent_transitions):
    """
    Function to filter out the silent transitions defined by a list from a given dictionary.
    :param dic: dictionary to be filtered, type: dictionary
    :param silent_transitions: list of silent transitions in an ocel log, type: list
    :return updated_dictionary: filtered dictionary, type: dictionary
    """
    updated_dictionary = {}
    for key, values in dic.items():
        if key not in silent_transitions:
            new_values = [val for val in values if val not in silent_transitions]
            updated_dictionary[key] = new_values
    return updated_dictionary

#recursive implementation of a depth-first search (DFS) algorithm
def dfs(graph, visited, activity, preceding_events):
    """
    Function to perform a depth-first search (DFS) algorithm on the activity graph.
    :param graph: activity graph, type: dictionary
    :param visited: set of already visited nodes, type: set
    :param activity: current activity, type: string
    :param preceding_events: list to store the preceding events, type: list
    """
    #takes as input the activity graph (represented as a dictionary), a set of visited nodes, the current activity, and a list to store the preceding events.
    visited.add(activity)
    for preceding_event in graph[activity]:
        #eighboring activity has not been visited yet, the algorithm visits it by calling the dfs function with the neighboring activity as the current activity.
        if preceding_event not in visited:
            dfs(graph, visited, preceding_event, preceding_events)
    preceding_events.append(activity)


def negative_events_with_weighting(ocel, ocpn):
    """
    Function to calculate the negative events measure with weighting based on the used places inside an object-centric petri-net.
    :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
    :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
    :return generalization: final value of the formula, type: float rounded to 4 digits
    """
    # since the process execution mappings have lists of length one,
    # we create another dictionary that only contains the the value inside the list to be able to derive the case
    mapping_dict = {key: ocel.process_execution_mappings[key][0] for key in ocel.process_execution_mappings}
    # we generate a new column in the class (log) that contains the process execution (case) number via the generated dictionary
    ocel.log.log['event_execution'] = ocel.log.log.index.map(mapping_dict)
    # generate a list of unique events in the event log
    events = np.unique(ocel.log.log.event_activity)
    # dictionary to store each activity as key and a list of its prior states/places as value
    targets = {}
    # dictionary to store each activity as key and a list of its following states/places as value
    sources = {}
    for arc in tqdm(ocpn.arcs, desc="Check the arcs"):
        # for each arc, check if our target is a valid transition
        if arc.target in ocpn.transitions:
            # load all the prior places of a valid transition into a dictionary, where the key is the transition and the value
            # a list of all directly prior places
            if arc.target.name in targets:
                targets[arc.target.name].append(arc.source.name)
            else:
                targets[arc.target.name] = [arc.source.name]
        if arc.source in ocpn.transitions:
            # load all the following places of a valid transition into a dictionary, where the key is the transition and the value
            # a list of all directly following places
            if arc.source.name in sources:
                sources[arc.source.name].append(arc.target.name)
            else:
                sources[arc.source.name] = [arc.target.name]
    # generate an empty dictionary to store the directly preceding transition of an activity
    preceding_activities = {}
    # use the key and value of targets and source to generate the dictionary
    for target_key, target_value in targets.items():
        preceding_activities[target_key] = []
        for source_key, source_value in sources.items():
            for element in target_value:
                if element in source_value:
                    preceding_activities[target_key].append(source_key)
                    break
    # generate an empty dictionary to store the directly succeeding transition of an activity
    succeeding_activities = {}
    for source_key, source_value in sources.items():
        succeeding_activities[source_key] = []
        for target_key, target_value in targets.items():
            for element in source_value:
                if element in target_value:
                    succeeding_activities[source_key].append(target_key)
                    break
    # store the name of all silent transitions in the log
    silent_transitions = [x.name for x in ocpn.transitions if x.silent]
    # replace the silent transitions in the succeeding activities dictionary by creating a new dictionary to store the modified values
    succeeding_activities_updated = {}
    # Iterate through the dictionary
    for key, values in succeeding_activities.items():
        # Create a list to store the modified values for this key
        new_values = []
        # Iterate through the values of each key
        for i in range(len(values)):
            # Check if the value is in the list of silent transitions
            if values[i] in silent_transitions:
                # Replace the value with the corresponding value from the dictionary
                new_values.extend(succeeding_activities[values[i]])
            else:
                # If the value is not in the list of silent transitions, add it to the new list
                new_values.append(values[i])
        # Add the modified values to the new dictionary
        succeeding_activities_updated[key] = new_values
    # create an empty dictionary to store all the preceding activities of an activity
    preceding_events_dict = {}
    # use a depth-first search (DFS) algorithm to traverse the activity graph and
    # create a list of all preceding events for each activity in the dictionary for directly preceding activities
    for activity in preceding_activities:
        # empty set for all the visited activities
        visited = set()
        # list for all currently preceding events
        preceding_events = []
        dfs(preceding_activities, visited, activity, preceding_events)
        # we need to remove the last element from the list because it corresponds to the activity itself
        preceding_events_dict[activity] = preceding_events[:-1][::-1]
    # delete all possible silent transitions from preceding_events_dict (dict where all direct preceding events are stored)
    filtered_preceding_events_full = filter_silent_transitions(preceding_events_dict, silent_transitions)
    # delete all possible silent transitions from filtered_preceding_events (dict where only direct preceding events are stored)
    filtered_preceding_events = filter_silent_transitions(preceding_activities, silent_transitions)
    # delete all possible silent transitions from succeeding_activities_updated (dict where only direct preceding events are stored)
    filtered_succeeding_activities_updated = filter_silent_transitions(succeeding_activities_updated,
                                                                       silent_transitions)
    # generate a grouped df such that we can iterate through the log case by case (sort by timestamp to ensure the correct process sequence)
    grouped_df = ocel.log.log.sort_values('event_timestamp').groupby('event_execution')
    DG = 0  # Disallowed Generalization initialisation
    AG = 0  # Allowed Generalization initialisation
    #create an empty list for all traces in the log
    trace_list = []
    for name, group in grouped_df:
        values = list(map(str, group['event_activity']))
        trace_list.append(values)
    #create an empty list for all the reversed suffixes
    suffix_lists = []
    for activities in trace_list:
        #generate the list of suffixes for the reversed traces for more efficient comparison
        reversed_activities = activities[::-1]
        suffixes = [reversed_activities[i:] for i in range(len(reversed_activities))]
        suffix_lists.append(suffixes)
    #Initialise the current trace index
    current_trace_index = 0
    # Iterate over each group
    for group_name, group_df in tqdm(grouped_df, total=len(grouped_df),
                                     desc="Calculate Generalization for all process executions"):
        # list for all the activities that are enabled, starting from all activities that do not have any preceding activity
        enabled = [key for key, value in filtered_preceding_events_full.items() if not value]
        # initialise a list of already executed activities in this trace
        trace = []
        # iterate through each case/process execution
        for index, row in group_df.iterrows():
            # Get the current negative events based on the current activity to be executed
            negative_activities = [x for x in events if x != row['event_activity']]
            # it may happen that an activity is not present in the model but nevertheless executed in the log
            if row['event_activity'] in enabled:
                # check which elements in the negative activity list are enabled outside of the current activity
                enabled.remove(row['event_activity'])
            # get all the negative events that can not be executed in the process model at the moment
            disallowed = [value for value in negative_activities if value not in enabled]
            # add activity that has been executed to trace
            trace.append(row['event_activity'])
            #only look at all other traces and the own trace after the current position
            modified_suffix_list = suffix_lists[:current_trace_index] + suffix_lists[current_trace_index+1:]
            #cut out the already played trace such that we start only at the position after the current event
            cutted_current_trace = trace_list[current_trace_index][len(trace):]
            #generate the list of suffixes for the reversed traces for more efficient comparison
            reversed_activities_current = cutted_current_trace[::-1]
            suffixes_current = [reversed_activities_current[i:] for i in range(len(reversed_activities_current))]
            #append the suffixes of the current trace without the position
            modified_suffix_list.append(suffixes_current)
            #initialise scoring dictionary to save the values
            activity_dict = {activity: [1] for activity in negative_activities}
            #for each negative event
            for activity in negative_activities:
                #set the window size first as the lenght of the trace - 1, because the current value has already been added to the trace
                window_size = len(trace)-1
                #initialise the matching_window
                matching_window = 0
                if window_size == 0:
                    #if the matching window is zero, break the loop, because we cannot compute a score (division by zero)
                    break
                #for each trace that is not the original trace
                for execution in modified_suffix_list:
                    #for each reversed suffix in the trace
                    for suffix in execution:
                            #set a counter for the moving window
                            l = 1
                            #while the counter is smaller than the window size or the remaining trace of the suffix and the values at the corresponding positions are the same
                            while (l < min(window_size,len(suffix)-1)) & (trace[len(trace)-l] == suffix[l-1]):
                                #increment the matching window by one position and the counter by one
                                matching_window = matching_window + 1
                                l = l + 1
                            #after the loop has been broken calculate the sccore and save it in the activity list
                            score = (window_size - matching_window)/window_size
                            activity_dict[activity].append(score)
            #initialise empty lists to store the minimum value of the lists for each activity
            min_values_enabled = []
            min_values_negative = []
            #loop over all events in the log
            for value in negative_activities:
                if value in activity_dict:
                    #compute the minimum value for eacch of the events outside of the current event
                    min_val = min(activity_dict[value])
                    #seperate the values that are enabled and disabled at this position
                    if value in enabled:
                        min_values_enabled.append(min_val)
                    elif value in disallowed:
                        min_values_negative.append(min_val)
            # update the values of allowed and disallowed generalizations based on the paper logic (incremend by 1-weight))
            AG = AG + (len(min_values_enabled)-sum(min_values_enabled))
            DG = DG + (len(min_values_negative)-sum(min_values_negative))
            # may happen that activities in the log are not in the process model
            if row['event_activity'] in filtered_succeeding_activities_updated:
                # get all possible new enabled activities
                possible_enabled = filtered_succeeding_activities_updated[row['event_activity']]
                # check if each activity has more than one directly preceding state
                for i in range(len(possible_enabled)):
                    # check if an event has two or more activities that need to be executed before the event can take place, if not add events to enabled
                    if len(filtered_preceding_events[possible_enabled[i]]) < 2:
                        enabled.append(possible_enabled[i])
                    # if all succeeding events equal all preceding events, we have a flower model and almost everything is enabled all the time
                    elif filtered_preceding_events[possible_enabled[i]] == filtered_succeeding_activities_updated[
                        possible_enabled[i]]:
                        enabled.append(possible_enabled[i])
                    else:
                        # if yes, check if all the needed activities have already been performed in this trace
                        if all(elem in trace for elem in filtered_preceding_events[possible_enabled[i]]):
                            enabled.append(possible_enabled[i])
            # extend the list with all elements that do not have any preceding activity and are therefore enabled anyways in our process model
            enabled.extend([key for key, value in filtered_preceding_events_full.items() if not value])
            # delete all duplicates from the enabled list
            enabled = list(set(enabled))
        #increate the current trace rat
        current_trace_index = current_trace_index + 1
    # calculate the generalization based on the paper
    generalization = AG / (AG + DG)
    return np.round(generalization, 4), np.round(AG, 4), np.round(DG, 4)

logging.basicConfig(filename='Negative_variants_runtime.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

filenames = ["BPI2017-Final"]
sample_sizes = [200 ,500, 800]
objects = [["application","offer"]]
counter = 0
for filename in filenames:
    object_types = objects[counter]
    counter += 1
    for sample_size in sample_sizes:
        logging.info(f"*** Negative Events {filename} Variant OCPN with {sample_size}**")
        ocel_variant = pd.read_pickle(f'/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/runtime/{filename}_{sample_size}_ocel_variant.pickle')
        logging.info(f"*** OCEL {filename}  with {sample_size}loaded**")

        with open(f"/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/runtime/{filename}_{sample_size}_variant_ocpn.pickle","rb") as file:
            variant_ocpn = pickle.load(file)

        logging.info(f"*** OCPN {filename}  with {sample_size}loaded**")

        start_time = time.time()
        value, AG, DG = negative_events_with_weighting(ocel_variant,variant_ocpn)
        execution_time = np.round(time.time() - start_time, 4)
        logging.info(f"The execution time for negative events generalization for {filename} and the variant with {sample_size} traces is {execution_time} seconds")