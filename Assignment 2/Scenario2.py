from FourRooms import FourRooms
import numpy as np
import matplotlib.pyplot as plt
import random

#Global variables
global T, directions
Qtable=np.zeros((144,4),dtype=float)
T=np.zeros((144,4),dtype=int) #R table
vis=np.zeros(144)
lr=0.1# Learning rate
numEpochs=100 #Number of epochs
discFactor=0.9 #Discount factor
epsilon=1 #Epsilon Greedy Strategy
packagesCollected=0# Packages collected
totalPackages = 3 #Total packages
beginningEpsilon=1
endEpsilon=numEpochs//2
epsilonDecayValue=epsilon/(endEpsilon-beginningEpsilon)
fourRoomsObj=FourRooms("multi")
directions = np.array([[0,-1],[0,1],[-1,0],[1,0]]) # UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 

#Function to populate R Table
def PopulateRTable() -> np.array([[int]]) :  
    #For populating the R table 
    for state in range(11*11):
        x,y = state%12 , state//12
        
        # Punishing the agent when it's taking an invalid move
        actSeq =[(x,y-1 if (y-1>=0 and  y-1< 12) else -1), # UP 
            (x, y+1 if( y+1>=0 and y+1 < 12) else -1), # DOWN
            (x-1 if (x-1>=0 and x-1<12) else -1,y), # LEFT 
            (x+1 if(x+1>=0 and x+1<12) else -1 ,y), # RIGHT
        ]
                
        for i,val in enumerate(actSeq):
            if val[0]<0 or val[1]<0:
                T[state,i] = -1
    return T
    
#For moving from one state to a new state given (state,action)
def Move(state:int ,action: int) -> int:
    x,y = state%12, state//12 # get the coordinates given a state 
    newState = [x,y] + directions[action]
    newState = np.array(newState) # make this a numpy array
    return newState[1]*12 + newState[0] # the new state s'
    
# for getting a list of possible moves given a state 
def PossibleMoves(state: int) -> [int] :
    actSeq = []
    for i,value in enumerate(T[state]):
        if value !=-1 :
            actSeq.append(i)
    return actSeq

#updating the rewards (T table) based on the agent's movement in the environment.
def rewards(prevState: int, newState: int, action: int, cell: int) -> int:
    if prevState == newState:  # If the agent hits an inner wall
        T[newState, action] = -100  # Punish the agent by setting a negative reward
        return T[prevState, action]
    else:
        T[prevState, action] = cell  # Update the reward for the current state-action pair
        return T[prevState, action] + 100  # Reward the agent with a positive reward of 100

def Qlearning(vis: np.array([[int]]), ep: int):
    x, y = fourRoomsObj.getPosition()
    state = y * 12 + x  # convert to 1D state
    end = False
    global epsilon
    packageCount = 0  # Variable to keep track of collected packages

    while not end:
        vis[state] += 1
        lr = 1 / (1 + vis[state])
        stateAction = PossibleMoves(state)

        if random.random() < epsilon:  # Explore
            action = stateAction[np.random.randint(0, len(stateAction))]
        else:
            action = np.argmax(Qtable[state])  # Exploit

        gridCell, currentPos, gType, isTerminal = fourRoomsObj.takeAction(action)

        rewards(state, currentPos[1] * 12 + currentPos[0], action, gridCell)
        nextState = currentPos[1] * 12 + currentPos[0]

        previousValue = Qtable[state, action]
        nextMax = np.max(Qtable[nextState])

        newValue = (1 - lr) * previousValue + lr * (gridCell + discFactor * nextMax)
        Qtable[state, action] = newValue

        end = isTerminal
        state = currentPos[1] * 12 + currentPos[0]

        # Check if a package was collected
        if gType == packageCount:
            packageCount += 1

        # Check if all packages have been collected
        if packageCount == 3:
            end = True

        if endEpsilon >= ep and ep >= beginningEpsilon:
            epsilon -= epsilonDecayValue

 
def ExploitEnvironment():
    x, y = fourRoomsObj.getPosition()
    state = y * 11 + x
    end = False
    packageCount = 0  # Variable to keep track of collected packages

    while not end:
        action = np.argmax(Qtable[state])

        _, currentPoss, gType, isTerminal = fourRoomsObj.takeAction(action)

        state = currentPoss[1] * 12 + currentPoss[0]

        # Check if a package was collected
        if gType == packageCount:
            packageCount += 1

        # Check if all packages have been collected
        if packageCount == 3:
            end = True

    fourRoomsObj.showPath(-1)
    plt.savefig("Scenario2.png")


def main():
    # Creating FourRooms Object and Populating R table with experience
    R = PopulateRTable()
    for e in range(0,numEpochs):
        fourRoomsObj.newEpoch()
        Qlearning(vis,e)

    fourRoomsObj.newEpoch()
    ExploitEnvironment()
  
if __name__ == "__main__":
    main()
      
     