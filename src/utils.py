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

# This python file serves as storage for functions that can be used throughout the thesis

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
    for variant in ocel.variants:
        # for each variant filter the log on all the cases belonging to this variant
        #filtered = ocel.log.log[ocel.log.log.event_variant.apply(lambda x: n in x)]
        # save the pandas df to a csv file such that we can reload it as object-centric log
        filename = f"{save_path_logs}{n}.csv"
        #filtered.to_csv(filename)
        parameters = {"obj_names": object_types,
                  "val_names": [],
                  "act_name": "event_activity",
                  "time_name": "event_timestamp",
                  "sep": ","}
        ocel_new = ocel_import_factory_csv.apply(file_path=filename, parameters=parameters)
        ocpn_new = ocpn_discovery_factory.apply(ocel_new, parameters={"debug": False})
        # append all the variant petri nets to our predefined list
        ocpn_nets.append(ocpn_new)
        n = n + 1
    #define empty lists for the final sets of arcs, places, and transitions for the final petri net
    Arcs = []
    Places =[]
    Transitions = []
    # for every petri net in our list
    for i in range(len(ocpn_nets)):
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
    # we generate the final object-centric petri net with our lists of places, transitions, and arcs
    variant_ocpn = ObjectCentricPetriNet(places = Places, transitions = Transitions, arcs = Arcs)
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