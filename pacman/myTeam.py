# myTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Raise
from typing import List, Tuple

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os, copy
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action
from tools import Queue
from tools import Search_node
from tools import manhattan_heuristic


CLOSE_DISTANCE = 5
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########                                       

class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    QLWeights = {
            "offensiveWeights":{'closest-food': -1, 
                                        'bias': 1, 
                                        '#-of-ghosts-1-step-away': -100, 
                                        'successorScore': 100, 
                                        'eats-food': 10, 
                                        'stop': -100},
            "defensiveWeights": {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2},
            "escapeWeights": {'onDefense': 10000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}
        }
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}


    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = True # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1 #default exploration prob, change to take a random step
        self.alpha = 0.1 #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists(MixedAgent.QLWeightsFile):
            with open(MixedAgent.QLWeightsFile, "r") as file:
                MixedAgent.QLWeights = eval(file.read())
            print("Load QLWeights:",MixedAgent.QLWeights )
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        print("Write QLWeights:", MixedAgent.QLWeights)
        file = open(MixedAgent.QLWeightsFile, 'w')
        file.write(str(MixedAgent.QLWeights))
        file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action (up down left right wait) to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects,initState)

        # Check if we can stick to current plan
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan
            print("\tPLAN:",self.highLevelPlan)
        if len(self.highLevelPlan)==0:
            raise Exception("Solver retuned empty plan, you need to think how you handle this situation or how you modify your model ")
        
        # Get next action from the plan
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print("\tAgent:",self.index, MixedAgent.CURRENT_ACTION[self.index])
        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState, highLevelAction) #Generate low level plan with heuristic search planner
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex+=1
        print("\tAgent ", self.index ,':',lowLevelAction)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)
        
        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None:
            if cloestFoodDist <=CLOSE_DISTANCE:
                states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning_gt",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            #collect ally's state(action)
            if agent_index != self.index:
                if MixedAgent.CURRENT_ACTION[agent_index] == "fallback" or MixedAgent.CURRENT_ACTION[agent_index] == "defend_enemy" or MixedAgent.CURRENT_ACTION[agent_index] == "fallback_together" or MixedAgent.CURRENT_ACTION[agent_index] == "chase_enemy_at_home" or MixedAgent.CURRENT_ACTION[agent_index] == "eat_enemy_at_home":
                    temp = "a{}".format(agent_index)
                    states.append(("defend",temp))    
                elif MixedAgent.CURRENT_ACTION[agent_index] == "go_to_enemy_land":
                    temp = "a{}".format(agent_index)
                    states.append(("go_enemy_land",temp))    
                elif MixedAgent.CURRENT_ACTION[agent_index] == "go_home" or MixedAgent.CURRENT_ACTION[agent_index] == "go_home_unpack" or MixedAgent.CURRENT_ACTION[agent_index] == "unpack_food":
                    temp = "a{}".format(agent_index)
                    states.append(("go_home",temp))    
                elif MixedAgent.CURRENT_ACTION[agent_index] == "eat_capsule":
                    temp = "a{}".format(agent_index)
                    states.append(("eat_capsule",temp))    
                elif MixedAgent.CURRENT_ACTION[agent_index] == "eat_food":
                    temp = "a{}".format(agent_index)
                    states.append(("eat_food",temp))    
            
            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))
                
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
            

        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1
            
        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        
        if self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        if (("winning_gt10",) in initState):
            return self.goalDefWinning(objects, initState)
        else:
            return self.goalScoring(objects, initState)

    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.

        positiveGoal = []
        negtiveGoal = [("food_available",)] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            if agent_type == "current_agent":
                negtiveGoal += [("food_in_backpack", agent_obj)] # we have to unpack food at home to gain score.
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.
        
        return positiveGoal, negtiveGoal

    def goalDefWinning(self,objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 5 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "win_the_game" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        
        return positiveGoal, negtiveGoal

    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.
        
        map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState) # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        enemy_position = [enemy_state.getPosition() for enemy_index, enemy_state in enemies]
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents:
            if agent_index != self.index:
                ally_position = agent_state.getPosition()
        myPos = gameState.getAgentPosition(self.index)
        eat_food_loc = []
        def_food_loc = []
        entrance_loc = []
        exit_loc = [] 
        mid_line =  gameState.data.layout.width // 2
        # find the entracne exit and food location base on team
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                if foods[x][y] == True:
                    eat_food_loc.append((x,y))
                if foodNeedDefend[x][y] == True :  
                    def_food_loc.append((x,y))
        for y in range(0,gameState.data.layout.height):
            if gameState.isOnRedTeam(self.index):

                if not gameState.getWalls()[mid_line + 1 ][y]:
                    entrance_loc.append((mid_line + 1,y))
                if not gameState.getWalls()[mid_line - 1 ][y]:
                    exit_loc.append((mid_line - 1,y))
            else:
                if not gameState.getWalls()[mid_line - 1][y]:
                    entrance_loc.append((mid_line - 1,y))
                if not gameState.getWalls()[mid_line + 1 ][y]:
                    exit_loc.append((mid_line + 1,y))
        #action "go_to_enemy_land"
        if highLevelAction == "go_to_enemy_land" :
            dis_entrance = []
            block_list = [i for i in enemy_position]
            #get the enemy location if possible
            if  any(i is not None for i in enemy_position):
                #avoid that entrance near enemies
                temp_distance = [[],[]]
                for a in enemy_position:
                    if a is not None:
                        for entrance in entrance_loc:
                            temp_distance[enemy_position.index(a)].append(self.getMazeDistance(a, (entrance[0],entrance[1])))
                            
                avoid_entrance = []
                temp = []
                for d in temp_distance:
                    temp_entrance = copy.deepcopy(entrance_loc)
                    if len(d)>0:
                        temp.append(temp_entrance.pop(d.index(min(d))))
                for i in temp:        
                    if temp in entrance_loc:
                        entrance_loc.pop(entrance_loc.index(temp))
            #find the nearest entrance
            temp_entrance_a = copy.deepcopy(entrance_loc)
            for entrance in entrance_loc:
                dis_entrance.append(self.getMazeDistance(myPos, entrance))
            target = entrance_loc[dis_entrance.index(min(dis_entrance))] 
            temp_entrance_a.pop(temp_entrance_a.index(target))
            block_list = block_list + temp_entrance_a
            #get path 
            path = self.get_path(gameState,myPos,target,block_list)
            #get next action
            for legal_action in gameState.getLegalActions(self.index):
                if path[1] == Actions.getSuccessor(myPos,legal_action):
                    return [(legal_action,path[1])]
        #action "eat_food"
        elif highLevelAction == "eat_food":
            block_list = []
            # get the block list. when do the path finding,try to avoid the enemis position
            for i in enemy_position:
                if i != None:
                    block_list.append(i)
            # get the rest food location
            if len(eat_food_loc) >0:
                dis_food = [self.getMazeDistance(myPos,loc)  for loc in eat_food_loc]
                target_food = eat_food_loc[dis_food.index(min(dis_food))]
            # get path 
                path = self.get_path(gameState,myPos, target_food,block_list)
                for legal_action in gameState.getLegalActions(self.index):
                    if path[1] == Actions.getSuccessor(myPos,legal_action):
                        return [(legal_action,path[1])]
            # no food avalible just stay there
            else:
                for legal_action in gameState.getLegalActions(self.index):
                    if myPos == Actions.getSuccessor(myPos,legal_action):
                        return [(legal_action,myPos)]
        # action "go_home" "unpack_food" "go_home_unpack" "go_home_to_defend"
        elif highLevelAction == "go_home" or highLevelAction == "unpack_food" or  highLevelAction == "go_home_unpack" or highLevelAction == 'go_home_to_defend':
            dis_exit = []
            target = []
            block_list = []
            temp_exit = copy.deepcopy(exit_loc)
            # get enemies position if the distance between agent and exit is longer than the distance between enemies and exit, this exit point will be saved in block list  
            if  any(i is not None for i in enemy_position):
                #avoid that entrance
                enemy_ent_distance = [[],[]]
                agent_ent_distance = []
                for a in enemy_position:
                    if a is not None:
                        block_list.append(a)
                        # check the distance between enemies and exit point
                        for entrance in exit_loc:
                            enemy_ent_distance[enemy_position.index(a)].append(self.getMazeDistance(a, (entrance[0],entrance[1])))
                avoid_entrance = []
                for d in enemy_ent_distance:
                    if len(d)>0:
                        if self.getMazeDistance(myPos, (temp_exit[d.index(min(d))][0],temp_exit[d.index(min(d))][1])) > min(d): 
                            block_list.append(temp_exit[d.index(min(d))])
                            if temp_exit[d.index(min(d))] in exit_loc:
                                exit_loc.pop(exit_loc.index(temp_exit[d.index(min(d))]))
            #find the nearest exit point
            for exit_point in exit_loc:
                dis_exit.append(self.getMazeDistance(myPos, exit_point))
            target = exit_loc[dis_exit.index(min(dis_exit))]  
            for i in exit_loc:
                if i != target:
                    block_list.append(i)
            #get path 
            path = self.get_path(gameState,myPos,target,block_list)
            #get next action
            for legal_action in gameState.getLegalActions(self.index):
                if path[1] == Actions.getSuccessor(myPos,legal_action):
                    return [(legal_action,path[1])]
        # action "chase_enemy_at_home" "eat_enemy_at_home"
        elif highLevelAction == "chase_enemy_at_home" or highLevelAction == "eat_enemy_at_home":
            # if enemise position is avalible just go to the nearest enemy
            if  any(i is not None for i in enemy_position) and any(b[1].isPacman for b in enemies) :
                temp_distance = []
                for a in enemy_position:
                    if a is not None and enemies[enemy_position.index(a)][1].isPacman:
                        temp_distance.append(self.getMazeDistance(myPos, a))
                if len(temp_distance) > 1:
                    target = enemy_position[temp_distance.index(min(temp_distance))]
                else:
                    for a in enemy_position:
                        if a is not None:
                            target = a
            # else go for random food
            elif len(def_food_loc) > 0:
                target = random.choice(def_food_loc)
            # or go for random exit loc
            else:
                target = random.choice(exit_loc)
            #get path      
            path = self.get_path(gameState,myPos,target,[])
            # get next aaction
            for legal_action in gameState.getLegalActions(self.index):
                if path[1] == Actions.getSuccessor(myPos,legal_action):
                    return [(legal_action,path[1])]   
        # action 'fallback' 'fallback_together' 'patrol' 
        elif highLevelAction == "fallback" or highLevelAction == "fallback_together" or highLevelAction == "patrol":
            # stay near the entrance point and go up down and back
            if gameState.isOnRedTeam(self.index):
                if myPos[1] + 1 <= gameState.data.layout.height and not gameState.getWalls()[myPos[0]][myPos[1] + 1]:
                    target = (myPos[0],myPos[1] + 1) 
                elif myPos[1] - 1 >= 0 and not gameState.getWalls()[myPos[0]][myPos[1] - 1]:
                    target = (myPos[0],myPos[1] - 1)
                elif not gameState.getWalls()[myPos[0] - 1 ][myPos[1]]:
                    target = (myPos[0] - 1 ,myPos[1])
                path = self.get_path(gameState,myPos,target,[])
                for legal_action in gameState.getLegalActions(self.index):
                    if path[1] == Actions.getSuccessor(myPos,legal_action):
                        return [(legal_action,path[1])]
            else:
                if not gameState.getWalls()[myPos[0] + 1 ][myPos[1]]:
                    target = (myPos[0] + 1 ,myPos[1])
                elif myPos[1] + 1 <= gameState.data.layout.height and not gameState.getWalls()[myPos[0]][myPos[1] + 1]:
                    target = (myPos[0],myPos[1] + 1) 
                elif myPos[1] - 1 >= 0 and not gameState.getWalls()[myPos[0]][myPos[1] - 1]:
                    target = (myPos[0],myPos[1] - 1)    
            #get path   
                path = self.get_path(gameState,myPos,target,[])
                for legal_action in gameState.getLegalActions(self.index):
                    if path[1] == Actions.getSuccessor(myPos,legal_action):
                        return [(legal_action,path[1])]  
        # action 'action' 'take_a_detour_no_ally 'go_to_enemy_land_one_defend''go_to_enemy_land_one_defend_enemy_attack'
        elif highLevelAction == "take_a_detour" or  highLevelAction == "take_a_detour_no_ally" or  highLevelAction == "go_to_enemy_land_one_defend" or highLevelAction == "go_to_enemy_land_one_defend_enemy_attack":
            dis_entrance = []
            temp_entrance = copy.deepcopy(entrance_loc)
            #choose the upper or downer entrance base on all or enemies
            for i in enemy_position:
                if i != None:
                    if i[1] < (gameState.data.layout.height//2):
                        target = max(entrance_loc)
                    else:
                        target = min(entrance_loc)
                    break
                else:
                    if ally_position[1] < (gameState.data.layout.height//2):
                        target = max(entrance_loc)
                    else:
                        target = min(entrance_loc)
                    break
            temp_entrance.pop(target)  
            # path find      
            path = self.get_path(gameState,myPos,target,temp_entrance)
            # get next action
            for legal_action in gameState.getLegalActions(self.index):
                if path[1] == Actions.getSuccessor(myPos,legal_action):
                    return [(legal_action,path[1])]       
        # other action
        else:
            if  any(i is not None for i in enemy_position):
                temp_distance = []
                for a in enemy_position:
                    if a is not None:
                        temp_distance.append(self.getMazeDistance(myPos, a))
                target = enemy_position[temp_distance.index(min(temp_distance))]
            else:
                temp_distance = []
                for entrance in exit_loc:
                    temp_distance.append(self.getMazeDistance(myPos, entrance))
                target = exit_loc[temp_distance.index(min(temp_distance))]
            #get path   
            path = self.get_path(gameState,myPos,target,[])
            for legal_action in gameState.getLegalActions(self.index):
                if path[1] == Actions.getSuccessor(myPos,legal_action):
                    return [(legal_action,path[1])]            
         # You should return a list of tuple of move action and target location (exclude current location).
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True
    
    def get_path(self,gameState: GameState,start: tuple, goal: tuple,block: list):
        ############
        # Below is  path finding implementation,
        ############
        map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        #enemy_position = [enemy_state.getPosition() for enemy_index, enemy_state in enemies]
        #enemy_state = [enemy_state.scaredTimer for enemy_index, enemy_state in enemies]
        path = []
        block_list = []
        for i in block:
            if i != None:
                block_list.append((float(i[0]),float(i[1]))) 
        
        queue_temp = Queue()
        closed_list = []
        start_node = Search_node(state = start, goal = goal, parent = None)
        #add the start node to the queue
        queue_temp.push(start_node)
        # getting the satrt node
        search_node = queue_temp.pop() 
        while search_node != None and search_node.state_ != goal:
            closed_list.append(search_node)
            # get all valid moveabel direction node
            for action in  Actions._directions:
                nextPos = Actions.getSuccessor(search_node.state_,action)
                # try to block the node is wall or in block list(enemies and other entrance)
                if map[int(nextPos[0])][int(nextPos[1])] == False:
                    if nextPos in block_list:
                        search_node.g_ = sys.maxsize
                        search_node.h_ = sys.maxsize
                        node = Search_node(nextPos,goal,search_node)
                        node.action_ = action  
                        if not node in search_node.child:
                            search_node.child.append(node) 
                    else: 
                        node = Search_node(nextPos,goal,search_node)
                        node.action_ = action  
                        if not node in search_node.child:
                            search_node.child.append(node)    
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
    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0
        if highLevelAction == "go_to_enemy_land" or highLevelAction == "eat_food":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home" or highLevelAction == "unpack_food":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update 

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                            # print("Agent",self.index," weights:", weights)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    def getOffensiveReward(self, gameState, nextState):
        # Calculate the reward. NEEDS WORK
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        dis_list = []
        #keep away from enemy
        for enemy in enemies:
            if not enemy[1].isPacman:
                if not enemy[1].getPosition() == None:
                    dis_list.append(self.getMazeDistance((int(nextAgentState.getPosition()[0]),int(nextAgentState.getPosition()[1])),(int(enemy[1].getPosition()[0]),int(enemy[1].getPosition()[1]))))
        reward = 0
        if len(dis_list) > 0 and min(dis_list) < MEDIUM_DISTANCE and nextAgentState.isPacman: #only cares about ghosts if its a pacman and the distance is less than three
            reward -= min(dis_list)
        #unpack food reward
        new_food_carry = nextAgentState.numCarrying - currentAgentState.numCarrying
        #new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        if new_food_carry > 0:
            #get new food, return positive scre
            reward += 200
        elif new_food_carry < 0:
            # agent is eaten! by enemy! return large negative reward.
            reward +=  -200
        else:
            # nothing happens
            reward +=  -1
        mid_line =  gameState.data.layout.width // 2
        if not gameState.isOnRedTeam(self.index):
            home = mid_line - 1
        else:
            home = mid_line + 1

        home_loc = []
        if nextAgentState.isPacman:
            for y in range(0,gameState.data.layout.height):
                if not gameState.getWalls()[home][y]:
                    home_loc.append((home,y))   
            mindis = min([self.getMazeDistance((int(nextAgentState.getPosition()[0]),int(nextAgentState.getPosition()[1])),loc) for loc in home_loc])
            reward += mindis
        
        return reward
    def getDefensiveReward(self,gameState, nextState):
        print("Warnning: DefensiveReward not implemented yet, and learnning rate is 0 for defensive ",file=sys.stderr)
        return 0
    
    def getEscapeReward(self,gameState, nextState):
        print("Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",file=sys.stderr)
        return 0



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState, action):
        food = self.getFood(gameState) 
        walls = gameState.getWalls()
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        
        # Initialize features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(successor)

        # Bias
        features["bias"] = 1.0
        
        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
                features["eats-food"] = 1.0

        # Number of Ghosts scared
        #features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist

        # Normalize and return
        features.divideAll(5.0)
        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    


    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]
    


    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    

