# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        '''
            Code for printing game state 
        '''
        # print("Successor Game State:", successorGameState)
        # print("New Pos:", newPos)
        # print("New Food:", newFood.asList())
        # # print("New Ghost States:", newGhostStates)
        # print("New Scared Times:", newScaredTimes)
        # print("New Actions:", action)
        # print('---------------------------------------')

        if currentGameState.isWin() or currentGameState.isLose():
            return 0
        #Walls and Current foods
        curFood = currentGameState.getFood()
        walls = currentGameState.getWalls().asList()

        # get pos of foods
        food_list = newFood.asList()
        cur_food_list = curFood.asList()

        # eva score
        score = 0
        if action is Directions.STOP:
            return -1

        # ghost distances
        ghost_distances = []
        avg_ghost_distance,sum = 0,0
        for ghost_state in newGhostStates:
            print("Current ghost position:",ghost_state.getPosition())
            ghost_distances.append(manhattanDistance(newPos, ghost_state.getPosition()))

        # print("legal actions:", successorGameState.getLegalActions())
        # for distance in ghost_distances:
        #     print("ghost distance",distance)
        #     sum += distance
        # avg_ghost_distance = sum/len(ghost_distances)

        # scared time
        for scaretime in newScaredTimes:
            score += scaretime

        # food distances
        food_distances = []
        for food_position in food_list:
            food_distances.append(manhattanDistance(newPos,food_position))

        # evaluation
        inverse_food_distances = 0
        if len(food_distances)>0 and min(food_distances)>0:
            inverse_food_distances = 1.0/min(food_distances)
        #score += max(ghost_distances) * pow(inverse_food_distances,3)
        # score += min(ghost_distances) * pow(inverse_food_distances,4)
        score += avg_ghost_distance * pow(inverse_food_distances,4)
        # print("Function eva:",avg_ghost_distance * pow(inverse_food_distances,5))
        score += successorGameState.getScore()
        # print("Gamestate score:",successorGameState.getScore())

        if newPos in cur_food_list:
            score = score * 1.1
            # print("Final eva score(new pos in current food):",score)

        # print("Final score:",score)
        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agent_index):
            MAX, MIN= float('Inf'), float('-Inf')
            # check agent type
            # print("Agent number:",state.getNumAgents())
            if agent_index >= state.getNumAgents():
                agent_index = 0
                depth = depth + 1
            # check depth and state
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # Pacman
            if agent_index == 0:
                max_result = ['', MIN]
                # get pacman actions
                for pacman_action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index, pacman_action)
                    result = minimax(successor, depth, agent_index + 1)
                    if type(result) is not list:
                     max_score = result
                    else:
                        max_score = result[1]
                    if max_score > max_result[1]:
                        max_result = [pacman_action, max_score]
                return max_result
            # Ghost
            elif agent_index > 0:
                min_result = ['', MAX]
                for ghost_action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index, ghost_action)
                    result = minimax(successor, depth, agent_index + 1)
                    if type(result) is not list:
                        min_score = result
                    else:
                        min_score = result[1]
                    if min_score < min_result[1]:
                        min_result = [ghost_action, min_score]
                return min_result
        result = minimax(gameState,0,0)
        print(result)
        return result[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        MAX, MIN = float('Inf'), float('-Inf')
        alpha, beta = MIN,[]
        current_alpha = MIN
        result = ''
        for element in range(1,gameState.getNumAgents()):
            beta.append(MAX)
        def alphaBetaPruning(state, depth, alpha, beta):
            # check depth and state
            agent_num = state.getNumAgents()
            if state.isWin() or state.isLose() or depth == self.depth * agent_num:
                return self.evaluationFunction(state)
            # check agent type
            agent_index = depth%agent_num
            if agent_index == 0:
                betas = beta[:]
                alpha_value = MIN
                for action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index,action)
                    alpha_value = max(alpha_value, alphaBetaPruning(successor, depth+1, alpha, betas))
                    if alpha_value > min(betas):
                        return alpha_value
                    alpha = max(alpha_value, alpha)
                return alpha_value
            elif agent_index >0:
                betas = beta[:]
                beta_value = MAX
                for action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index,action)
                    beta_value = min(beta_value, alphaBetaPruning(successor, depth+1, alpha, betas))
                    if beta_value < alpha:
                        return beta_value
                    betas[agent_index-1] = min(beta_value,betas[agent_index-1])
                return beta_value

        for action in gameState.getLegalActions(0):
            alpha_result = alphaBetaPruning(gameState.generateSuccessor(0,action),1,alpha,beta)
            if alpha_result > current_alpha:
                current_alpha = alpha_result
                result = action
            alpha = max(alpha, current_alpha)
        print(result)
        return result
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent_index):
            MAX, MIN = float('Inf'), float('-Inf')
            # check agent type
            # print("Agent number:",state.getNumAgents())
            if agent_index >= state.getNumAgents():
                agent_index = 0
                depth = depth + 1
            # check depth and state
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # Pacman
            if agent_index == 0:
                max_result = ['', MIN]
                # get pacman actions
                for pacman_action in state.getLegalActions(agent_index):
                    successor = state.generateSuccessor(agent_index, pacman_action)
                    result = expectimax(successor, depth, agent_index + 1)
                    if type(result) is not list:
                        max_score = result
                    else:
                        max_score = result[1]
                    if max_score > max_result[1]:
                        max_result = [pacman_action, max_score]
                return max_result
            # Ghost
            elif agent_index > 0:
                min_result = ['', MAX]
                sum_of_score, sum_of_actions = 0,0
                for ghost_action in state.getLegalActions(agent_index):
                    sum_of_actions +=1
                    successor = state.generateSuccessor(agent_index, ghost_action)
                    result = expectimax(successor, depth, agent_index + 1)
                    if type(result) is not list:
                        score = result
                    else:
                        score = result[1]
                    sum_of_score += score
                expect_value = float(sum_of_score)/float(sum_of_actions)
                expect_action = random.sample(state.getLegalActions(agent_index),1)
                return [expect_action, expect_value]
        result = expectimax(gameState,0,0)
        # print(result)
        return result[0]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''
    No Question 5
    '''
    pass
    util.raiseNotDefined()
# Abbreviation
better = betterEvaluationFunction

