from copy import deepcopy
from lib_piglet.search.search_node import Search_node
from lib_piglet.utils.data_structure import Queue
from lib_piglet.utils.tools import eprint
import glob, os, sys , math
try:
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

def a_star(start: tuple, start_direction: int, goal: tuple,local_env: RailEnv,agent_id:int,constraints):
    """ 
        start   - start position
        start_direction
        goal    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    open_list = Queue()
    closed_list = []
    start_node = Search_node(state = start, direction = start_direction, goal = goal, parent = None)
    open_list.push(start_node)
    curr = open_list.pop()
    while curr != None and curr.state_ != goal:
        closed_list.append(curr)
        # get all valid node traversable from the current node in current node's children
        curr = transition_nodes(node=curr, env=local_env)
        if (curr.state_,curr.timestep_) in constraints:
            continue
        else:
            for child in curr.child:
                if (child.state_ , child.timestep_) not in constraints:
                    if not child in closed_list:
                        new_node = deepcopy(child)
                        new_node.parent_ = curr
                        new_node = conflict_detection(new_node, constraints) 
                        open_list.push(new_node)
        curr = open_list.pop()
    path = get_path_node(curr)
    return path  # Failed to find solutions


def get_path_node(node):
    path = []
    curr = node
    while curr is not None:
        path.append(curr.state_)
        curr = curr.parent_
    path.reverse()
    return path

def transition_nodes(node:Search_node, env):
    # get available transitions from Rail_Env object.
    valid_transitions = env.rail.get_transitions(node.state_[0], node.state_[1], node.action_)
    # find valid position and direction for the agent for the current rail environment
    for i in range(0, len(valid_transitions)):
        if valid_transitions[i]:
            new_x = node.state_[0]
            new_y = node.state_[1]
            timestep = node.timestep_
            action = i
            if action == Directions.NORTH:
                    new_x -= 1
            elif action == Directions.EAST:
                    new_y += 1
            elif action == Directions.SOUTH:
                    new_x += 1
            elif action == Directions.WEST:
                    new_y -= 1
            # add new node to the current node's children
            new_node = Search_node((new_x,new_y),action, node.goal_,node) 
            node.child.append(new_node)   
    new_node = Search_node(node.state_,node.action_, node.goal_,node)      
    if not new_node in node.child:
        node.child.append(new_node)
    return node

def get_path(start: tuple, start_direction: int, goal: tuple, env: RailEnv):
    path = a_star(start, start_direction, goal,env,0,[])
    return path


def conflict_detection(node, constrains):
    conflict = False
    for p in constrains:
        if p[1] -1 == node.timestep_ and p[0] == node.state_:
            conflict = True
        if p[1]  == node.timestep_ and p[0] == node.state_:
            conflict = True
        if p[1] + 1 == node.timestep_ and p[0] == node.state_:
            conflict = True
    if conflict:
        node.h_ = math.inf
        node.f_ = node.g_ + node.h_
    return node

