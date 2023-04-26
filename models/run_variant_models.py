import warnings
warnings.filterwarnings('ignore')

import pickle
import logging
import numpy as np
from tqdm import tqdm
import os
import re

#function to filter out the silent transitions defined by a list from a given dictionary
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

def generate_variant_model(ocel,save_path_logs,object_types,save_path_visuals = None):
    """
    Function to generate the variant model of an JSONOCEL-log, return it and save it as svg.
    :param ocel: given OCEL-log, type: OCEL-Log
    :param save_path_logs: path for the saved variant logs, type: string
    :param object_types: list of object types that are present in the log, type: list
    :param save_path_visuals: path for the saved variant model visualization, type: string
    :return: variant model, type: object-centric petri net
    """
    #list to save the variant nets
    ocpn_nets = []
    n = 0 # running variable for number of variants
    for variant in tqdm(ocel.variants, desc="Generating Variant Models"):
        # for each variant filter the log on all the cases belonging to this variant
        filtered = ocel.log.log[ocel.log.log.event_variant.apply(lambda x: n in x)]
        # save the pandas df to a csv file such that we can reload it as object-centric log
        filename = f"{save_path_logs}{n}.csv"
        filtered.to_csv(filename)
        parameters = {
            "obj_names": object_types,
            "val_names": [],
            "act_name": "event_activity",
            "time_name": "event_timestamp",
            "sep": ",",
        }
        ocel_new = ocel_import_factory_csv.apply(file_path=filename, parameters=parameters)
        ocpn_new = ocpn_discovery_factory.apply(ocel_new, parameters={"debug": False})
        # append all the variant petri nets to our predefined list
        ocpn_nets.append(ocpn_new)
        n += 1
        # define empty lists for the final sets of arcs, places, and transitions for the final petri net
    Arcs = []
    Places = []
    Transitions = []
    # for every petri net in our list
    for i in tqdm(range(len(ocpn_nets)), desc="Processing Variant Nets"):
        # first check the places if they are initial or final places
        for place in ocpn_nets[i].places:
            if (place.initial == True) | (place.final == True):
                # Find the number at the end of the string of the intial/final places and swap them with inital/final respectively
                match = re.search(r'\d+$', place.name)
                if match:
                    # Get the matched string and strip it from the original string
                    matched_number = match.group()
                    if (place.initial == True):
                        place.name = f"{place.name.rstrip(matched_number)}_initial"
                        Places.append(place)
                    if (place.final == True):
                        place.name = f"{place.name.rstrip(matched_number)}_final"
                        Places.append(place)
            else:
                # if not just append the current count of the variant to the place name such that we can distinguish them
                place.name = f"{place.name}_{i}"
                Places.append(place)
        # for all transitions append the current count of the variant to the place name such that we can distinguish them
        for transition in ocpn_nets[i].transitions:
            transition.name = f"{transition.name}_{i}"
            Transitions.append(transition)
        # add all the arcs to our final set, we do not need to care about the names anymore because these are adopted from the transition and place definitions
        for arc in ocpn_nets[i].arcs:
            Arcs.append(arc)
    print('#########Start generating Object-Centric Petri Net#########')
    # we generate the final object-centric petri net with our lists of places, transitions, and arcs
    variant_ocpn = ObjectCentricPetriNet(places = Places, transitions = Transitions, arcs = Arcs)
    print('#########Finished generating Object-Centric Petri Net#########')
    return variant_ocpn

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


def negative_events_without_weighting(ocel, ocpn):
    """
    Function to calculate the negative events measure without weighting based on the used places inside an object-centric petri-net.
    :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
    :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
    :return generalization: final value of the formula, type: float rounded to 4 digits
    """

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
    # generate an empty dictionary to store the directly preceeding transition of an activity
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
    # create an empty dictionary to store all the precedding activities of an activity
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
    # delete all possible silent transitions from preceding_events_dict (dict where all direct preceeding events are stored)
    filtered_preceeding_events_full = filter_silent_transitions(preceding_events_dict, silent_transitions)
    # delete all possible silent transitions from filtered_preceeding_events (dict where only direct preceeding events are stored)
    filtered_preceeding_events = filter_silent_transitions(preceding_activities, silent_transitions)
    # delete all possible silent transitions from succeeding_activities_updated (dict where only direct preceeding events are stored)
    filtered_succeeding_activities_updated = filter_silent_transitions(succeeding_activities_updated,
                                                                       silent_transitions)
    # generate a grouped df such that we can iterate through the log case by case (sort by timestamp to ensure the correct process sequence)
    grouped_df = ocel.log.log.sort_values('event_timestamp').groupby('event_execution')
    DG = 0  # Disallowed Generalization intialisation
    AG = 0  # Allowed Generalization intialisation
    # Iterate over each group
    for group_name, group_df in tqdm(grouped_df, total=len(grouped_df),
                                     desc="Calculate Generalization for all process executions"):
        # Iterate over each row in the group
        # list for all the activities that are enabled, starting from all activities that do not have any preceeding activity
        enabled = [key for key, value in filtered_preceeding_events_full.items() if not value]
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
            # update the values of allowed and disallowed generalizations based on the paper logic
            AG = AG + len(enabled)
            DG = DG + len(disallowed)

            # may happen that activities in the log are not in the process model
            if row['event_activity'] in filtered_succeeding_activities_updated:
                # get all possible new enabled activities
                possible_enabled = filtered_succeeding_activities_updated[row['event_activity']]
                # check if each activity has more than one directly preceeding state
                for i in range(len(possible_enabled)):
                    # check if an event has two or more activities that need to be executed before the event can take place, if not add events to enabled
                    if len(filtered_preceeding_events[possible_enabled[i]]) < 2:
                        enabled.append(possible_enabled[i])
                    # if all succeeding events equal all preceeding events, we have a flower model and almost everything is enabled all the time
                    elif filtered_preceeding_events[possible_enabled[i]] == filtered_succeeding_activities_updated[
                        possible_enabled[i]]:
                        enabled.append(possible_enabled[i])
                    else:
                        # if yes, check if all the needed activities have already been performed in this trace
                        if all(elem in trace for elem in filtered_preceeding_events[possible_enabled[i]]):
                            enabled.append(possible_enabled[i])
            # extend the list with all elements that do not have any preceeding activity and are therefore enabled anyways in our process model
            enabled.extend([key for key, value in filtered_preceeding_events_full.items() if not value])
            # delete all duplicates from the enabled list
            enabled = list(set(enabled))
    # calculate the generalization based on the paper
    generalization = AG / (AG + DG)
    return np.round(generalization, 4)

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
logger.info('This log file shows the results of the variant model negative events for the BPI log')
filename_variant = "/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS3_variant_log.csv"
object_types = ["incident","customer"]
parameters = {"obj_names": object_types,
              "val_names": [],
              "act_name": "event_activity",
              "time_name": "event_timestamp",
              "sep": ","}
ocel_variant = ocel_import_factory_csv.apply(file_path=filename_variant, parameters=parameters)

filename = "/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/jsonocel/DS3.jsonocel"
ots = ["incident","customer"]
ocel = ocel_import_factory.apply(filename)
variant_ocpn = generate_variant_model(ocel,save_path_logs='../src/data/csv/DS3_variants/DS3_variant',object_types = ots)


value = negative_events_without_weighting (ocel_variant, variant_ocpn)
print(value)
logger.info("*** Evaluate ***")
logger.info('The value of generalization for DS3 is %s', value)