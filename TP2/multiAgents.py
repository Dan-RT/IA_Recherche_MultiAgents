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
import time
from pacman import SCARED_TIME

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

        score = 0

        #If a ghost is present return a score of 0
        if newPos in [ghostState.getPosition() for ghostState in newGhostStates]:
          return 0
        
        #The distance to the closest food is substracted to the score
        if len(newFood.asList()) != 0:
          successorDistanceToClosestFood = min([manhattanDistance(newPos, foodPosition) for foodPosition in newFood.asList()])
          score -= successorDistanceToClosestFood

        #The distance to dangerous ghosts is added to the score
        ghostPacmanDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        for i in range(len(ghostPacmanDistances)):
          if newScaredTimes[i] <= ghostPacmanDistances[i]:
            score += ghostPacmanDistances[i]

        return score+successorGameState.getScore()


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
            Returnsy the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.max_pacman(gameState, 0, True)

    def max_pacman(self, state, depth, surface):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        best_utility = float("-inf")
        utility_tmp = best_utility
        best_action = 0

        possible_actions = state.getLegalActions(0)

        for action in possible_actions:
            utility_tmp = self.min_ghost(state.generateSuccessor(0, action), depth, 1)
            if utility_tmp > best_utility:
                best_utility = utility_tmp
                best_action = action

        if surface:
            return best_action
        else:
            return best_utility

    def min_ghost(self, state, depth, ghost):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if state.getNumAgents()-1 == ghost:
            # pacman is next
            next_ghost = 0
        else:
            next_ghost = ghost+1

        best_utility = float("inf")
        current_utility = best_utility

        possible_actions = state.getLegalActions(ghost)
        #print(possible_actions)

        for action in possible_actions:
            if next_ghost == 0:
                if depth != self.depth-1:
                    current_utility = self.max_pacman(state.generateSuccessor(ghost, action), depth+1, False)
                else:
                    current_utility = self.evaluationFunction(state.generateSuccessor(ghost, action))
            else:
                current_utility = self.min_ghost(state.generateSuccessor(ghost, action), depth, next_ghost)

            best_utility = MIN(best_utility, current_utility)
        return best_utility


def MAX(val1, val2):
    if val1 >= val2:
        return val1
    else:
        return val2


def MIN(val1, val2):
    if val1 <= val2:
        return val1
    else:
        return val2


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        return self.max_pacman_alpha_beta(gameState, 0, True, float("-inf"), float("inf"))

    def max_pacman_alpha_beta(self, state, depth, surface, alpha, beta):
        #if surface:
            #print("\n\n-----------------------------------------\n")
        #print("\nMAX, depth: " + str(depth))

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        best_utility = float("-inf")
        utility_tmp = best_utility
        best_action = 0

        possible_actions = state.getLegalActions(0)
        #print(possible_actions)

        for action in possible_actions:
            utility_tmp = self.min_ghost_alpha_beta(state.generateSuccessor(0, action), depth, 1, alpha, beta)
            if utility_tmp > best_utility:
                best_utility = utility_tmp
                best_action = action
            if best_utility > beta:
                #print("\n\nReturning action: " + best_action + " - utility: " + str(best_utility))
                return best_utility
            alpha = MAX(alpha, best_utility)
        if surface:
            #print("\n\nReturning action: " + best_action + " - utility: " + str(best_utility))
            return best_action
        else:
            return best_utility

    def min_ghost_alpha_beta(self, state, depth, ghost, alpha, beta):

        #print("MIN, ghost: " + str(ghost) + " depth: " + str(depth))

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if state.getNumAgents()-1 == ghost:
            # pacman is next
            next_ghost = 0
        else:
            next_ghost = ghost+1

        best_utility = float("inf")
        current_utility = best_utility

        possible_actions = state.getLegalActions(ghost)
        #print(possible_actions)

        for action in possible_actions:
            if next_ghost == 0:
                if depth != self.depth-1:
                    current_utility = self.max_pacman_alpha_beta(state.generateSuccessor(ghost, action), depth+1, False, alpha, beta)
                else:
                    current_utility = self.evaluationFunction(state.generateSuccessor(ghost, action))
            else:
                current_utility = self.min_ghost_alpha_beta(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
            best_utility = MIN(best_utility, current_utility)
            if best_utility < alpha:
                return best_utility
            beta = MIN(beta, best_utility)
        return best_utility


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def max_pacman_expectimax(self, state, depth, surface):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        best_utility = float("-inf")
        utility_tmp = best_utility
        best_action = 0

        possible_actions = state.getLegalActions(0)

        for action in possible_actions:
            utility_tmp = self.min_ghost_expectimax(state.generateSuccessor(0, action), depth, 1)
            if utility_tmp > best_utility:
                best_utility = utility_tmp
                best_action = action

        if surface:
            return best_action
        else:
            return best_utility

    def min_ghost_expectimax(self, state, depth, ghost):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if state.getNumAgents()-1 == ghost:
            # pacman is next
            next_ghost = 0
        else:
            next_ghost = ghost+1

        current_utility = 0 #float("inf")
        possible_actions = state.getLegalActions(ghost)
        nb_possible_actions = len(possible_actions)

        if nb_possible_actions == 0:
            return self.evaluationFunction(state)

        for action in possible_actions:
            if next_ghost == 0:
                if depth != self.depth-1:
                    current_utility = current_utility + self.max_pacman_expectimax(state.generateSuccessor(ghost, action), depth+1, False)
                else:
                    current_utility = current_utility + self.evaluationFunction(state.generateSuccessor(ghost, action))
            else:
                current_utility = current_utility + self.min_ghost_expectimax(state.generateSuccessor(ghost, action), depth, next_ghost)

        return current_utility / float(nb_possible_actions)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        return self.max_pacman_expectimax(gameState, 0, True)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = 0

    #If a ghost is present return a score of 0
    if currentGameState.isLose():
        return 0

    #If the state is a win returns a score of +inf
    if currentGameState.isWin():
        return float("inf")
    
    numberOfFoodLeft = len(food.asList())
    averageDistanceTofood = 0
    hasFood = False
    if pos in food.asList():
        hasFood = True
    #The distance to the closest food is substracted to the score
    if len(food.asList()) != 0:
        distanceToClosestFood = min([manhattanDistance(pos, foodPosition) for foodPosition in food.asList()])
        averageDistanceTofood = sum([manhattanDistance(pos, foodPosition) for foodPosition in food.asList()]) / float(len(food.asList()))
    

    numberOfDangerousGhosts = 0
    distancesToDangerousGhosts = []
    #The distance to dangerous ghosts is added to the score
    ghostPacmanDistances = [manhattanDistance(pos, ghostState.getPosition()) for ghostState in ghostStates]
    for i in range(len(ghostPacmanDistances)):
        if scaredTimes[i] <= ghostPacmanDistances[i]:
            numberOfDangerousGhosts += 1
            distancesToDangerousGhosts.append(ghostPacmanDistances[i])
            distancesToDangerousGhosts.sort()
            if distancesToDangerousGhosts[0] < 2:
                score = 0
            else:
                score = distancesToDangerousGhosts[0] 

    score -= distanceToClosestFood*3 - averageDistanceTofood + random.randint(1,3)
    if hasFood:
        score = score * 1.2 


    return score+currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(AlphaBetaAgent):
  """
    Your agent for the mini-contest
  """
      
  def __init__(self):
      pass
  
  def getAction(self, gameState):
      util.raiseNotDefined()