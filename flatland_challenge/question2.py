"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""
# TODO: flatland virtual env check. debug mode option.

from audioop import add
from ctypes.wintypes import MAX_PATH
from lib_piglet.utils.tools import eprint
from lib_piglet.search.search_node import Search_node
from lib_piglet.heuristics.gridmap_h import manhattan_heuristic
from question1 import transition_nodes
import glob, os, sys,math,time
from lib_piglet.utils.data_structure import Queue
from copy import deepcopy


#import necessary modules that this python scripts need.
try:
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 2

#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param env The flatland railway environment
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, local_env: RailEnv, agent_id: int, existing_paths: list):
    ############
    # Below is an dummy path finding implementation,
    # which always choose the first available transition of current state.
    #
    # Replace these with your implementation and return a list of (x,y) tuple as your plan.
    # Your plan should avoid conflicts with paths in existing_paths.
    ############

    # initialize path list
    time_start = time.time()
    queue = Queue()                             # queue for storing nodes
    saft_interval_dict= {}               # dictionary to saving saft interval nodes
    #getting the longest path
    max_len = 0
    for p in existing_paths:
        if len(p) > max_len:
            max_len = len(p)
    # creating nodes with safe intervals for the start node
    start_node = Search_node(state = start, direction = start_direction, goal = goal, parent = None)
    node = start_node
    safe_interval = conflict_detection(node, node.timestep_, max_len * 2, existing_paths)
    # only keep the start node with safe interval starting at 0
    key = unique_key(loc=node.state_, direction=node.action_,env=local_env)
    saft_interval_dict [key] = safe_interval

    for node in safe_interval:
        if node.state_ == start and node.interval[0] == 0:
            queue.push(node)

    search_node = queue.pop()

    # iterate until queue becomes exhausted or goal is reached
    while search_node != None and search_node.state_ != search_node.goal_:
        current_time = time.time()
        if current_time -  time_start >= 5:
            break
        # get all valid node traversable
        search_node = transition_nodes(node=search_node, env=local_env)
        for child in search_node.child:
            node = child
            # create a unique key
            key = unique_key(loc=node.state_, direction=node.action_,env=local_env)
            # save the safe interval
            if key in saft_interval_dict:
                safe_interval = saft_interval_dict[key]
            else:
                safe_interval = conflict_detection(node, node.timestep_, max_len * 2, existing_paths)
                saft_interval_dict[key] = safe_interval
            #check the child 
            for node in safe_interval:
                    # if the child nodes safe interval does match with the current node
                if search_node.interval[1] + 1 < node.interval[0] or search_node.interval[0] + 1 > node.interval[1]:
                    continue
                # update for the current node
                new_node = deepcopy(node)
                new_node.parent_ = search_node
                if new_node.interval[0] < search_node.interval[0] + 1:
                    new_node.interval[0] = search_node.interval[0] + 1
                    new_node.timestep_ = search_node.interval[0] + 1
                    new_node.g_ = search_node.interval[0] + 1
                    new_node.f_ = new_node.g_ + new_node.h_ 
                queue.push(new_node)
        search_node = queue.pop()
    if search_node != None:
        path =[search_node.state_]
    else:
        path= []
        return path
    while search_node.parent_!= None:
        i = search_node.interval[0] - search_node.parent_.interval[0]
        while i > 0:
            path.append(search_node.parent_.state_)
            i -= 1
        search_node = search_node.parent_
    path.reverse()
    return path

def conflict_detection(node, time, max_len, existing_paths):
    relax_location = []    # safe intervals
    node.parent_ = None   # removing any relation with other nodes
    # search for the next timesteps
    for i in range(time, max_len):
        conflict = False
        for loc in existing_paths:
            # checking whether conflict with other agents before this timestep
            if 0 <= i - 1 < len(loc) and loc[i - 1] == node.state_:
                conflict = True
            # checking whether conflict with other agents in this timestep
            if 0 <= i < len(loc) and loc[i] == node.state_:
                conflict = True
            # checking whether conflict with other agents after this timestep   
            if 0 <= i + 1 < len(loc) and loc[i + 1] == node.state_:
                conflict = True
        if conflict:
            #if this node habe saft interval
            if node.interval[1] != math.inf:
                relax_location.append(node)
            # create new node when a conflict occurs
            node = Search_node(node.state_, node.action_, node.goal_)
        else:
            # if node don't have move interval
            if node.interval[1] == math.inf:
                #update the information
                node.interval[0] = i
                node.timestep_ = i
                node.g_ = i
                node.f_ = node.g_ + node.h_ 
            node.interval[1] = i
    relax_location.append(node)
    return relax_location

def unique_key(loc, direction,env):
        return str((loc[0] * env.width) + loc[1]) + str(direction)


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"multi_test_Case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_Case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,2)


















