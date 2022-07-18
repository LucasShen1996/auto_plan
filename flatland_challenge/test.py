from cbs import CBSSolver
from lib_piglet.utils.tools import eprint
import glob, os, sys,time,json
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
debug = True
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
    method_cbs = CBSSolver(local_env)
    path_all =  method_cbs.find_solution()
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



