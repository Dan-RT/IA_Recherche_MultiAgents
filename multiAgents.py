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

    def minTurn(self, state, currentDepth, agent):
        #If the state is a leaf, its score is returned
        if state.isLose() or state.isWin():
          return state.getScore()

        #Since there is possibly several ghosts (min players) it is needed to check 
        #if the next player is a max player (pacman) or a min player (a ghost)
        nextAgent = agent + 1
        if state.getNumAgents() - 1 == agent:
          nextAgent = 0

        bestUtility = float("inf")
        utility = bestUtility

        #All possible actions of the ghost are explored
        actions = state.getLegalActions(agent)
        for action in actions:

          #If the next player is pacman maxTurn is called, else it is another ghost's turn so minTurn is called
          if nextAgent == 0:
            if currentDepth != self.depth-1:
              utility = self.maxTurn(state.generateSuccessor(agent, action), currentDepth + 1)
            else:
              utility = self.evaluationFunction(state.generateSuccessor(agent, action))
          else:
            utility = self.minTurn(state.generateSuccessor(agent, action), currentDepth, nextAgent)

          if bestUtility > utility:
            bestUtility = utility
        return bestUtility

    def maxTurn(self, state, currentDepth):
        #If the state is a leaf, its score is returned
        if state.isWin() or state.isLose():
          return state.getScore()

        bestUtility = float("-inf")
        utility = bestUtility

        #All possible actions of pacman are explored
        actions = state.getLegalActions(0)
        for action in actions:
          #After pacman's turn it is always a ghost's turn
          utility = self.minTurn(state.generateSuccessor(0, action), currentDepth, 1)

          if utility > bestUtility:
            bestUtility = utility
            bestAction = action

        if currentDepth != 0:
          return bestUtility
        else:
          return bestAction

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
        #Pacman (max player) is the first one to execute an action
        return self.maxTurn(gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

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