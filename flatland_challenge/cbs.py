try:
    from flatland.envs.rail_env import RailEnv
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:

    exit(1)

from lib_piglet.search.search_node import Search_node
from question1 import transition_nodes
from lib_piglet.utils.data_structure import Queue


class CBSPlanner():
    # initial parameter
    def __init__(self,start,start_direction,goal,agent_id,env,existing_path):
        self.env = env
        self.start = start
        self.start_direction = start_direction
        self.goal = goal
        self.agent_id = agent_id
        self.open = Queue()
        if len(existing_path) != 0:
            self.exsisting_path_list = self.get_conflict_list(existing_path)
        else:
            self.exsisting_path_list = []
    # get valid next step
    def get_successors(self, node:Search_node):
        successors = []
        curr_node = transition_nodes(node,self.env)
        for child in curr_node.child:
            # conflict check
            if len(self.exsisting_path_list) != 0:
                conflict = False
                for path in self.exsisting_path_list:
                    for node in path:
                        if child == node:
                            conflict = True
                if conflict == False:
                    successors.append(child)    
            else:
                successors.append(child)
        return successors
    # compute the path plan
    def compute_plan(self):
        self.open.clear()
        goal_reached = False
        start_node = Search_node(state = self.start,direction= self.start_direction,goal = self.goal,parent=None)
        self.open.push(start_node)
        while (not goal_reached):
            if len(self.open.queue) == 0: 
                # Plan not found
                return []
            current_node = self.open.pop()
            # get next vaild step
            successors = self.get_successors(current_node)
            for successor in successors:
                    if successor.state_ == self.goal:
                        goal_reached = True
                        goal_node = successor
                        break
                    #push vaild next step to the open list
                    self.open.push(successor)
        # Tracking back
        return  goal_node
    
    # create a location tuple path       
    def get_path(self,node:Search_node):
        path_list = []
        while node.parent_!=None:
            path_list.append(node.state_)
            node = node.parent_
        path_list.append(node.state_)
        path_list.reverse()
        return path_list
    # create a search node list existing_path
    def get_conflict_list (self,existing_path):
        busy = []
        if len(existing_path) >0:
            for i in range(len(existing_path)):
                busy.append([])
            agent_id = 0
            for path in existing_path:
                if len(path) > 0:
                    start_loc  = path[0]
                    goal_loc = path[-1]
                    direction = get_action(start_loc,path[1])
                    start_node = Search_node(start_loc,direction,goal_loc,None)
                    busy[agent_id].append(start_node)
                    for i in range(1,len(path)):
                        # the last node will have the same action with previous node 
                        if path[i] == path[-1] :
                            direction = get_action(path[i-1],path[i])
                            node = Search_node(path[i],direction,goal_loc,busy[agent_id][-1])
                        else:
                            direction = get_action(path[i],path[i+1])
                            # wait action will have the same action with previous node
                            if direction == 4:
                                node = Search_node(path[i],busy[agent_id][-1].action_,goal_loc,busy[agent_id][-1])
                            else:
                                node = Search_node(path[i],direction,goal_loc,busy[agent_id][-1])
                        busy[agent_id].append(node)
                agent_id += 1
        return busy
# direction detect for the exsisting path
def get_action(curr_loc: tuple,next_loc: tuple):
    if  curr_loc== next_loc:
        return 4
    move_direction = 0
    if next_loc[0] - curr_loc[0] == 1:
        move_direction = Directions.SOUTH
    elif next_loc[0] - curr_loc[0] == -1:
        move_direction = Directions.NORTH
    elif next_loc[1] - curr_loc[1] == -1:
        move_direction = Directions.WEST
    elif next_loc[1] - curr_loc[1] == 1:
        move_direction = Directions.EAST

        return move_direction