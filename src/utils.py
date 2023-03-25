from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from ocpa.objects.log.exporter.ocel import factory as ocel_export_factory
from ocpa.algo.util.filtering.log.variant_filtering import filter_infrequent_variants
from ocpa.visualization.oc_petri_net import factory as ocpn_vis_factory
from ocpa.objects.oc_petri_net.obj import ObjectCentricPetriNet
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
