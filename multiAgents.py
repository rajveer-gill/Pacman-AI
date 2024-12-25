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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        nearestFood = min(foodDistances) if foodDistances else 1
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhost = min(ghostDistances) if ghostDistances else 1
        foodScore = 10.0 / nearestFood 
        ghostPenalty = -200 if nearestGhost <= 1 else -10.0 / nearestGhost
        if any(newScaredTimes):
            ghostPenalty /= 2 
        return successorGameState.getScore() + foodScore + ghostPenalty

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max(minimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else:
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
        return max(gameState.getLegalActions(0),
                   key=lambda action: minimax(1, 0, gameState.generateSuccessor(0, action)))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(1, depth, successor, alpha, beta))
                    alpha = max(alpha, value)
                    if alpha > beta: 
                        break  
                return value
            else:  
                value = float('inf')
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
                    beta = min(beta, value)
                    if alpha > beta: 
                        break  
                return value
        bestAction = None
        alpha, beta = float('-inf'), float('inf')
        maxValue = float('-inf')
        for action in gameState.getLegalActions(0): 
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successor, alpha, beta)
            if value > maxValue:
                maxValue = value
                bestAction = action
            alpha = max(alpha, maxValue)

        return bestAction
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0: 
                return max(expectimax(1, depth, state.generateSuccessor(agentIndex, action))
                           for action in state.getLegalActions(agentIndex))
            else: 
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                actions = state.getLegalActions(agentIndex)
                probabilities = 1.0 / len(actions)
                return sum(expectimax(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)) * probabilities
                           for action in actions)
        return max(gameState.getLegalActions(0),
                   key=lambda action: expectimax(1, 0, gameState.generateSuccessor(0, action)))


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()
    foodDistance = min(manhattanDistance(pos, f) for f in food) if food else 1
    capsuleDistance = min(manhattanDistance(pos, c) for c in capsules) if capsules else 1
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghosts]
    scaredGhostDistances = [
        manhattanDistance(pos, ghost.getPosition())
        for ghost in ghosts if ghost.scaredTimer > 0
    ]
    activeGhostDistances = [
        manhattanDistance(pos, ghost.getPosition())
        for ghost in ghosts if ghost.scaredTimer == 0
    ]
    foodScore = 10.0 / foodDistance
    capsuleScore = 20.0 / capsuleDistance if capsules else 0 
    ghostPenalty = sum(15.0 / max(d, 1) for d in activeGhostDistances) 
    scaredBonus = sum(10.0 / max(d, 1) for d in scaredGhostDistances)
    survivalBonus = 100 if not currentGameState.isLose() else 0
    score = currentGameState.getScore() + foodScore + capsuleScore - ghostPenalty + scaredBonus + survivalBonus
    return score
better = betterEvaluationFunction
