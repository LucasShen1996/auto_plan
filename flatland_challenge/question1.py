"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""
from lib_piglet.utils.tools import eprint
import glob, os, sys,math
from lib_piglet.utils.data_structure import Queue
# from search_node import Search_node
from lib_piglet.search.search_node import Search_node
#import necessary modules that this python scripts need.
try:
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = True
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0


#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
#########################


# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param env The flatland railway environment
# @return path A list of (x,y) tuple.
def get_path(start: tuple, start_direction: int, goal: tuple, env: RailEnv):
    ############
    # Below is an dummy path finding implementation,
    # which always choose the first available transition of current state.
    # Replace these with your implementation and return a list of (x,y) tuple as your plan.
    # Your plan should avoid conflicts with paths in existing_paths.
    ############
    path = []
    queue_temp = Queue()
    closed_list = []
    start_node = Search_node(state = start, direction = start_direction, goal = goal, parent = None)
    #add the start node to the queue
    queue_temp.push(start_node)
    # getting the satrt node
    search_node = queue_temp.pop() 
    while search_node != None and search_node.state_ != goal:
        closed_list.append(search_node)
        # get all valid moveabel direction node 
        search_node = transition_nodes(node=search_node, env=env)
        for child in search_node.child:
            if not child in closed_list:
                queue_temp.push(child)
        # getting next node with(smallest F value)   
        search_node = queue_temp.pop()
    # creating a path
    while search_node!= None:
        path.append(search_node.state_)
        search_node = search_node.parent_
    path.reverse()
    return path


def transition_nodes(node:Search_node, env):
    # get available transitions from Rail_Env object.
    valid_transitions = env.rail.get_transitions(node.state_[0], node.state_[1], node.action_)
    # find valid position and direction for the agent for the environment
    for i in range(0, len(valid_transitions)):
        if valid_transitions[i]:
            new_x = node.state_[0]
            new_y = node.state_[1]
            action = i
            if action == Directions.NORTH:
                new_x -= 1
            elif action == Directions.EAST:
                new_y += 1
            elif action == Directions.SOUTH:
                new_x += 1
            elif action == Directions.WEST:
                new_y -= 1
            # add new node to the current node's child list
            new_node = Search_node((new_x,new_y),action,node.goal_,node)
            if not new_node in node.child:
                node.child.append(new_node)
    # add wating node in the the child list
    new_node = Search_node(node.state_,node.action_, node.goal_,node)     
    if not new_node in node.child:
        node.child.append(new_node)
    return node
#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)












