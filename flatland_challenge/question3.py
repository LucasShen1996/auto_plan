
from lib_piglet.utils.tools import eprint
import glob, os, sys,time,json
import question1
import question2
from copy import deepcopy

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
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 0
test = 0

#########################
# Reimplementing the content in get_path() function.
#
# Return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
#########################

# This function return a list of location tuple as the solution.
# @param env The flatland railway environment
# @return path A list of (x,y) tuple.
def get_path(local_env: RailEnv):
    ############
    # Below is an dummy path finding implementation,
    # which always choose the first available transition of current state.
    #
    # Replace these with your implementation and return a list of paths. Each path is a list of (x,y) tuple as your plan.
    # Your plan should avoid conflicts with each other.
    ############

    # initialize path list
    path_all = []
    for i in range(len(local_env.agents)):
        path_all.append([])
    # creating the list of agents   
    agent_list = list(range(len(local_env.agents)))

    # iterate over all agents in the agent list
    idx = 0
    while idx < len(agent_list):
        # getting agent and its starting postion, direction and goal position
        agent = agent_list[idx]
        start_loc = local_env.agents[agent].initial_position
        start_direction = local_env.agents[agent].initial_direction
        goal = local_env.agents[agent].target
        # setting planner based on planner type
        node_list = question2.get_path(start=start_loc, start_direction=start_direction, goal=goal, local_env=local_env,agent_id= agent, \
            existing_paths=path_all)
        # finding the node list from the planner
        path_all[agent] = []
        for curr_node in node_list:
            path_all[agent].append(curr_node)
        idx += 1
        # if the path is empty arrange the agent list such that it puts the conflicting agent at the first
        # and start from the beginning by reinitialising list of all paths
        if path_all[agent] == []:
            # creating new agent list by putting conflicting agent at the start of the list
            new_agent_list = [agent]
            for curr_agent in agent_list:
                if curr_agent != agent:
                    new_agent_list.append(curr_agent)
            agent_list = deepcopy(new_agent_list)
            # reinitialising list of all paths
            path_all = []
            for _ in agent_list:
                path_all.append([])
            idx = 0
    return path_all




#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_Case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3,deadline_files)
        




