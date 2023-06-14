import numpy as np
from tqdm import tqdm
from src.utils import filter_silent_transitions, dfs


# Define your function that updates the global variable
def update_global_variables_without(group, filtered_preceding_events_full, filtered_preceding_events,
                            filtered_succeeding_activities_updated, events, silent_transitions, AG, DG):
    """
    Function to update the global variables AG and DG based on the paper logic.
    :param group: group of the event log that is currently processed, type: pandas dataframe
    :param filtered_preceding_events_full: dictionary that contains all preceding events for each activity, type: dictionary
    :param filtered_preceding_events: dictionary that contains all directly preceding events for each activity, type: dictionary
    :param filtered_succeeding_activities_updated: dictionary that contains all succeeding activities for each activity, type: dictionary
    :param events: list of all activities in the process model, type: list
    :param silent_transitions: list of all silent transitions in the process model, type: list
    :param AG: global variable for allowed generalizations, type: int
    :param DG: global variable for disallowed generalizations, type: int
    :return: updated values for AG and DG, type: int
    """
    # Iterate over each row in the group
    # list for all the activities that are enabled, starting from all activities that do not have any preceding activity
    enabled = [key for key, value in filtered_preceding_events_full.items() if not value]
    # initialise a list of already executed activities in this trace
    trace = []
    # iterate through each case/process execution
    for index, row in group.iterrows():
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
    return AG, DG

# Define the function that will be executed in parallel
def process_group_without(args):
    """
    Function to process a group of the event log in parallel
    :param args: set of variables for the measure calculation (see original function)
    :return: updated values for AG and DG, type: int
    """
    group_key, df_group, filtered_preceding_events_full, filtered_preceding_events, \
    filtered_succeeding_activities_updated, events, silent_transitions, AG, DG = args
    AG, DG = update_global_variables_without(df_group, filtered_preceding_events_full, filtered_preceding_events,
                                     filtered_succeeding_activities_updated, events, silent_transitions, AG, DG)
    return AG, DG

def negative_events_without_weighting_parallel(ocel, ocpn):
    """
    Function to calculate the variables for negative events measure paralle calculation without weighting
    :param ocel: object-centric event log for which the measure should be calculated, type: ocel-log
    :param ocpn: corresponding object-centric petri-net, type: object-centric petri-net
    :return: set of variables for the measure calculation (see original function)
    """
    # since the process execution mappings have lists of length one,
    # we create another dictionary that only contains the  value inside the list to be able to derive the case
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

    return grouped_df, filtered_preceding_events_full, filtered_preceding_events, filtered_succeeding_activities_updated, events, silent_transitions
