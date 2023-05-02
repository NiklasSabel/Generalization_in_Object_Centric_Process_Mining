from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory
from ocpa.algo.util.filtering.log.variant_filtering import filter_infrequent_variants
from ocpa.visualization.oc_petri_net import factory as ocpn_vis_factory
from ocpa.objects.oc_petri_net.obj import ObjectCentricPetriNet
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
import re
import numpy as np
import os
from tqdm import tqdm
import random

# This python file serves as storage for functions that can be used throughout the thesis

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

def get_happy_path_log(load_path, save_path=None) :
    """
    Function to generate the happy path log of an JSONOCEL-file and save it as JSONOCEL.
    :param load_path: path to the original JSONOCEL-log, type: string
    :param save_path: path for the saved happy-path JSONOCEL-log, type: string
    :return: preprocessed event log including only the happy path, type: ocel-log
    """
    # import the log
    ocel = ocel_import_factory.apply(load_path)
    # filter down on the most frequent variant
    filtered = filter_infrequent_variants(ocel, np.max(ocel.variant_frequencies) - 0.01)
    if save_path is not None:
        #export the variant
        ocel_export_factory.apply(filtered, save_path)

    return filtered

def save_process_model_visualization(ocel, save_path) :
    """
    Function to generate the process model of an JSONOCEL-log and save it as svg.
    :param ocel: given OCEL-log, type: OCEL-Log
    :param save_path: path for the saved process model visualization, type: string
    """
    #change the environment path for visualization
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    # get the object-centric petri net
    ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})
    # generate the visualization
    gviz = ocpn_vis_factory.apply(ocpn, parameters={'format': 'svg'})
    # save the visualization
    ocpn_vis_factory.save(gviz, save_path)



def create_flower_model(load_path,ots,save_path=None):
    """
    Function to generate the flower model of an JSONOCEL-log, return it and save it as svg.
    :param load_path: path to the original JSONOCEL-log, type: string
    :param ots: list of object types that are present in the log, type: list
    :param save_path: path for the saved process model visualization, type: string
    :return: flower model, type: object-centric petri net
    """
    # change the environment path for visualization
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    #import the JSONOCEL-log and transform it into an object-centri petri net
    ocel = ocel_import_factory.apply(load_path)
    ocpn = ocpn_discovery_factory.apply(ocel, parameters={"debug": False})
    # define lists for the final set of arcs, transitions, and places of the flower model
    arcs = []
    transitions = []
    places = []
    # transform the list of object-types into a list of places of the object-centric petri net
    [places.append(ObjectCentricPetriNet.Place(name=c,object_type=c,initial=True)) for c in ots]
    #for each transition in our given object centric petri net create a transition for the flower model
    for t in [x for x in ocpn.transitions if not x.silent]:
        t_new = ObjectCentricPetriNet.Transition(name=t.name)
        transitions.append(t_new)
        # for each object type create arcs to every transition that it visits in the log
        for ot in ots:
            if ot in [a.source.object_type for a in t.in_arcs]:
                var = any([a.variable for a in t.in_arcs if a.source.object_type == ot ])
                source_place = [p for p in places if p.initial and p.object_type == ot][0]
                in_a = ObjectCentricPetriNet.Arc(source_place, t_new, variable = var)
                out_a = ObjectCentricPetriNet.Arc(t_new, source_place, variable = var)
                arcs.append(in_a)
                arcs.append(out_a)
                t_new.in_arcs.add(in_a)
                t_new.out_arcs.add(out_a)
    #create the flower model with the given set of places, transitions, and arcs
    flower_ocpn = ObjectCentricPetriNet(places = places, transitions = transitions, arcs = arcs)
    if save_path is not None:
        # generate a visualization of the flower model and save it as svg
        gviz = ocpn_vis_factory.apply(flower_ocpn, parameters={'format': 'svg'})
        ocpn_vis_factory.save(gviz, save_path)
    return flower_ocpn

def generate_variant_model(ocel,save_path_logs,object_types,save_path_visuals = None,save=None):
    """
    Function to generate the variant model of an JSONOCEL-log, return it and save it as svg.
    :param ocel: given OCEL-log, type: OCEL-Log
    :param save_path_logs: path for the saved variant logs, type: string
    :param object_types: list of object types that are present in the log, type: list
    :param save_path_visuals: path for the saved variant model visualization, type: string
    :param save: indicator if variant logs need to be saved again, if not generated before, type: string
    :return: variant model, type: object-centric petri net
    """
    #list to save the variant nets
    ocpn_nets = []
    n = 0 # running variable for number of variants
    for variant in tqdm(ocel.variants, desc="Generating Variant Models"):
        filename = f"{save_path_logs}{n}.csv"
        if save is not None:
            # for each variant filter the log on all the cases belonging to this variant
            filtered = ocel.log.log[ocel.log.log.event_variant.apply(lambda x: n in x)]
            # save the pandas df to a csv file such that we can reload it as object-centric log
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
    if save_path_visuals is not None:
        #change the environment path for visualization
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
        gviz = ocpn_vis_factory.apply(variant_ocpn, parameters={'format': 'svg'})
        ocpn_vis_factory.save(gviz, save_path_visuals)
    return variant_ocpn

def generate_variant_log(ocel, save_path, filtered = False):
    """
    Function to generate the variant log of an JSONOCEL-log, return it and save it as csv.
    :param ocel: given OCEL-log, type: OCEL-Log
    :param save_path: path for the saved variant log as csv file, type: string
    :param filtered: boolean value to indicate if the log was filtered before, type: boolean
    :return: variant log, type: pandas dataframe
    """
    if filtered == False:
        #generate the variant column by executing ocel.variants once only if the log was not filtered before, because then
        # we already have a variant column given
        ocel.variants
    # get a copy of df
    df = ocel.log.log.copy()
    # create a mapping dictionary of integer values to input strings
    if filtered == False:
        # if not filtered, the column incorporates a list
        values = sorted(set(val[0] for val in df['event_variant']))
    else:
        # if filtered, the column incorporates a string
        values = sorted(set(val[1] for val in df['event_variant']))
    mapping = {value: f'{value}' for value in values}
    # apply the mapping to the 'event_activity' column and concatenate with the 'event_variant' column to e.g., "Event_0" to
    #represent the nodes of our variant petri net
    if filtered == False:
        # if not filtered, the column incorporates a list
        df['event_activity'] = df.apply(lambda x: f"{x['event_activity']}_{mapping[x['event_variant'][0]]}", axis=1)
    else:
        # if filtered, the column incorporates a string
        df['event_activity'] = df.apply(lambda x: f"{x['event_activity']}_{mapping[x['event_variant'][1]]}", axis=1)
    df.to_csv(save_path)
    return df


#function to generate a sample from traces from an object-centric petri net
def sample_traces(ocel, ocpn, amount, length = None):
    """
    Function to generate a sample of traces from an object-centric petri net.
    :param ocel: given OCEL-log, type: OCEL-Log
    :param ocpn: given object-centric petri net, type: ObjectCentricPetriNet
    :param amount: amount of traces to be generated, type: int
    :param length: maximum length of the traces to be generated, if not given, gets generated as double the average length in th log, type: int
    :return: list of sampled traces in lists, type: list
    """
   # we create another dictionary that only contains the the value inside the list to be able to derive the case
    mapping_dict = {key: ocel.process_execution_mappings[key][0] for key in ocel.process_execution_mappings}
    # we generate a new column in the class (log) that contains the process execution (case) number via the generated dictionary
    ocel.log.log['event_execution'] = ocel.log.log.index.map(mapping_dict)
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
    filtered_succeeding_activities_updated = filter_silent_transitions(succeeding_activities_updated,silent_transitions)
    #get average length of an process execution in the original log
    # group by event_execution and count the number of rows in each group
    grouped = ocel.log.log.groupby('event_execution').count()

    # find the event_execution with the lowest and highest number of rows to filter out outliers that distort the length
    lowest = grouped['event_activity'].idxmin()
    highest = grouped['event_activity'].idxmax()

    # filter out the rows corresponding to the process executions with the lowest and highest number of rows to filter out outliers
    df_filtered = ocel.log.log[(ocel.log.log['event_execution'] != lowest) & (ocel.log.log['event_execution'] != highest)]

    # group by event_execution and count the number of rows in each group again
    grouped_filtered = df_filtered.groupby('event_execution').count()

    # calculate the average number of activities per process execution
    avg_activities = grouped_filtered.mean()['event_activity']
    if length == None:
        limit_length = np.round(2 * avg_activities).astype(int)
    else:
        limit_length = length
    #define an empty list for the event log
    event_log_sampled = []
    # store the name of all non-silent transitions in the log to check for variant model in if else statements
    non_silent_transitions = [x.name for x in ocpn.transitions if not x.silent]
    #sample the desired amount of traces
    for j in tqdm(range(amount), desc="Generate the traces"):
        #get a list of all activities that need to be executed before the process is finished
        end_activities = [key for key, value in filtered_succeeding_activities_updated.items() if not value]
        # if all succeeding events equal all preceeding events, we have a flower model and almost everything is enabled all the time
        if filtered_preceeding_events==filtered_succeeding_activities_updated:
            enabled = list(np.unique(ocel.log.log.event_activity))
        #check if one of the non-silent transitions ends with a number, then we have a variant model
        elif non_silent_transitions[0][-1].isdigit():
            #generate the variants
            ocel.variants
            # get the amount of variants in the log
            amount_variants = len(np.unique(ocel.log.log['event_variant']))
            # Generate a random integer between 0 and amount of variants -1 to generate the path we are using
            trace_number = random.randint(0, amount_variants-1)
            # Use a list comprehension to filter out end activities that don't end with the trace number, also checks string length to avoid matching numbers containing the target number.
            end_activities = [x for x in end_activities if x.endswith(str(trace_number)) and (len(x) == len(str(trace_number)) or not x[-len(str(trace_number))-1].isdigit())]
            #generate the list of enabled activities
            enabled = [key for key, value in filtered_preceeding_events_full.items() if not value]
            # Use a list comprehension to filter out enabled activities that don't end with the trace number, also checks string length to avoid matching numbers containing the target number.
            enabled = [x for x in enabled if x.endswith(str(trace_number)) and (len(x) == len(str(trace_number)) or not x[-len(str(trace_number))-1].isdigit())]
        else:
            # list for all the activities that are enabled, starting from all activities that do not have any preceeding activity
            enabled = [key for key, value in filtered_preceeding_events_full.items() if not value]
        # initialise a list of already executed activities in this trace
        trace = []
        # the maximum length of a trace is the double of the average trace length in the log
        for i in range(limit_length):
            # get a random activity from the enabled cases to add to the trace
            # check if there are any enabled activities
            if enabled:
                # generate a random index for an enabled activity
                idx = random.randint(0, len(enabled)-1)

                #get the current activity
                executed_activity = enabled[idx]

                # add the activity at the random index to the trace
                trace.append(executed_activity)

                # remove the activity from enabled activities
                enabled.remove(executed_activity)
            # get all possible new enabled activities
            possible_enabled = filtered_succeeding_activities_updated[executed_activity]
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
            #check if activity is an end activity
            if executed_activity in end_activities:
                #if yes get all activites that need to be executed beforehand
                preceeding_activities = filtered_preceeding_events_full[executed_activity]
                #delete these activites from the enabled list if a loop may be possible
                enabled = [x for x in enabled if x not in preceeding_activities]
            # delete all duplicates from the enabled list
            enabled = list(set(enabled))
            #check if all end activities have been performed and if end_activities is non empty
            if end_activities and all(x in trace for x in end_activities):
                break
        event_log_sampled.append(trace)
    return event_log_sampled