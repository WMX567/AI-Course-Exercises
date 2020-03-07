# pacmanAgents.py
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


from pacman import Directions
from game import Agent
from heuristics import *
import random
import math

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        
        actionList = []
        possible = state.getAllPossibleActions()
        for i in range(5):
            actionList.append(possible[random.randint(0,len(possible)-1)])

        bestScore = -30000
        bestActions = []
        stop_flag = False
        #tempState = state
        while stop_flag == False:
            tempState = state
            for i in range(5):
                if tempState.isWin() + tempState.isLose() == 0:
                    tempState = tempState.generatePacmanSuccessor(actionList[i])
                    if tempState == None:
                        stop_flag = True
                        break
                else:
                    break

            if stop_flag == True:
                break
        
            tempScore = gameEvaluation(state, tempState)
            if tempScore > bestScore:
                bestScore = tempScore
                bestActions = actionList[:]

            actionList = bestActions[:]
            for i in range(5):
                choice = random.randint(0, 1)
                if choice == 1:
                    actionList[i] = possible[random.randint(0,len(possible)-1)]

        return bestActions[0]
        

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    def crossover(self, population, parent1_index, parent2_index):
        choice = random.randint(1, 100)
        if choice >= 71 and choice <= 100:
            return population[parent1_index], population[parent2_index]
        child1 = []
        child2 = []
        for i in range(5):
            choice = random.randint(0, 1)
            if choice == 0:
                child1.append(population[parent1_index][i])
            else:
                child1.append(population[parent2_index][i])

        for i in range(5):
            choice = random.randint(0, 1)
            if choice == 0:
                child2.append(population[parent1_index][i])
            else:
                child2.append(population[parent2_index][i])
        return child1, child2

    def mutate(self, population, possible):
        for i in range(8):
            for j in range(5):
                choice = random.randint(1, 100)
                if choice <= 10:
                    population[i][j] = possible[random.randint(0,len(possible)-1)]
        return population

    def computeRank(self, fitness_scores):
        rank = []
        for i in range(len(fitness_scores)):
            index = fitness_scores[i][0]
            score = fitness_scores[i][1]
            sum = 0
            count = 0
            for j in range(len(fitness_scores)):
                if fitness_scores[j][1] == score:
                    sum += j + 1
                    count += 1.0
            rank.append([index, sum/count])
        return rank

    def parent_selection(self, rank):
        rank.sort(key=lambda x:x[1])
        choice = random.randint(1, 36)
        sum = 0
        index = -1
        for i in range(len(rank)):
            index = rank[i][0]
            index_rank = rank[i][1]
            sum += index_rank
            if sum >= choice:
                break
        return index


    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        population = []
        possible = state.getAllPossibleActions()
        for i in range(8):
            actionList = []
            for j in range(5):
                actionList.append(possible[random.randint(0,len(possible)-1)])
            population.append(actionList)

        stop_flag = False
        tempState = state
        while stop_flag == False:
            #Evaluate the population
            fitness_scores = []
            for i in range(8):
                for j in range(5):
                    if tempState.isWin() + tempState.isLose() == 0:
                        tempState = tempState.generatePacmanSuccessor(population[i][j])
                        if tempState == None:
                            stop_flag = True
                            break
                    else:
                        break
                if tempState != None:
                    fitness_scores.append([i, gameEvaluation(state, tempState)])
                    tempState = state
                else:
                    break

            if stop_flag == True:
                break
            #Computer ranks of population
            fitness_scores.sort(key=lambda x:x[1])
            rank = self.computeRank(fitness_scores)

            #Select parents and crossover
            new_population = []
            for i in range(4):
                parent1_index = self.parent_selection(rank)
                parent2_index = self.parent_selection(rank)
                child1, child2 = self.crossover(population, parent1_index, parent2_index)
                new_population.append(child1)
                new_population.append(child2)

            #Mutate
            population = self.mutate(new_population, possible)

        return population[random.randint(0,len(population)-1)][0]


class Node:

    def __init__(self, action, parent):
        self.action = action
        self.parent = parent
        self.score = 0
        self.visitTimes = 1
        self.isFullyExpanded = False
        self.children = []
        self.actionList = []


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts

    isContinue = True

    def registerInitialState(self, state):
        return;

    def isTerminal(self, state):
        if state.isWin() + state.isLose() == 0:
            return False
        else:
            return True

    def backup(self, node, score):
        while node != None:
            node.visitTimes = node.visitTimes + 1.0
            node.score = node.score + score
            node = node.parent
        return

    def bestChild(self, node):
        best = []
        bestScore = - 30000
        for child in node.children:
            tempScore = child.score / child.visitTimes
            tempScore += math.sqrt(2*math.log(node.visitTimes)/child.visitTimes)
            if tempScore > bestScore:
                bestScore = tempScore
                best = [child]
            elif tempScore == bestScore:
                best.append(child)
        return random.choice(best)

    def getState(self, state, actionList):
        for action in actionList:
            state = state.generatePacmanSuccessor(action)
            if state == None:
                self.isContinue = False
                return None
            if self.isTerminal(state):
                break
        return state

    def defaultPolicy(self, state):
        tempState = state
        for i in range(5):
            if self.isTerminal(tempState):
                break
            legal = tempState.getLegalPacmanActions()
            action = legal[random.randint(0,len(legal)-1)]
            tempState = self.getState(tempState, [action])
            if tempState == None:
                return None
        return gameEvaluation(state, tempState)

    def expansion(self, nodeState, node):
        tried_actions = []
        for child in node.children:
            tried_actions.append(child.action)
        legal = nodeState.getLegalPacmanActions()
        for action in legal:
            if action not in tried_actions:
                newChild = Node(action, node)
                newChild.actionList = node.actionList[:]
                newChild.actionList.append(action)
                node.children.append(newChild)
                break
        if len(node.children) == len(legal):
            node.isFullyExpanded = True
        return node.children[-1]

    def treePolicy(self, root, node):
        nodeState = root
        while self.isTerminal(nodeState) == False:
            if node.isFullyExpanded == False:
                return self.expansion(nodeState, node)
            else:
                node = self.bestChild(node)
                nodeState = self.getState(root, node.actionList)
                if self.isContinue == False:
                    break
        return node

    # GetAction Function: Called with every frame
    def getAction(self, state):
        node = Node(None, None)
        self.isContinue = True
        while self.isContinue:
            newNode = self.treePolicy(state, node)
            if newNode == None:
                break
            newState = self.getState(state, newNode.actionList)
            if newState == None:
                break
            score = self.defaultPolicy(newState)
            if score == None:
                break
            self.backup(newNode, score)

        bestActions = []
        maxVisitedTime = -30000
        for child in node.children:
            if child.visitTimes > maxVisitedTime:
                maxVisitedTime = child.visitTimes

        for child in node.children:
            if child.visitTimes == maxVisitedTime:
                bestActions.append(child.action)

        return random.choice(bestActions)
