In this RL our goal is to cater for different scenarios with increasing complexities collecting packages.

Files contained in this submission are:
Makefile:   Used to automate the build process and easily run the python scripts
            To install necessary requirements and the virtual environment- type "make"
            To run Scenario1.py type "make run1"
            To run Scenario2.py type "make run2"

Scenario1.py:   Imported necessary libraries and modules: FourRooms from a module called "FourRooms," numpy, matplotlib.pyplot, and random.
                Defined global variables: T, directions, Qtable, vis, lr, numEpochs, discFactor, epsilon, beginningEpsilon, endEpsilon, and epsilonDecayValue.
                Created an instance of the FourRooms class called fourRoomsObj.
                Defined a function called PopulateRTable to populate the R table (T) with rewards for each state-action pair.
                Define helper functions:
                    Move to calculate the new state given a current state and action.
                    PossibleMoves to get a list of possible actions from a given state.
                Define the rewards function to update the rewards (T table) based on the agent's movement in the environment.
                Define the Qlearning function that implements the Q-learning algorithm. It updates the Q-table (Qtable) based on the agent's interactions with the environment using epsilon-greedy exploration and exploitation.
                Define the ExploitEnvironment function to test the learned policy by exploiting the environment. It follows the actions with the highest Q-values until reaching a terminal state and visualizes the path taken.
                Define the main function that:
                    Populates the R table with rewards.
                    Executes the Q-learning algorithm for a specified number of epochs.
                    Exploits the environment to visualize the learned policy.
                Call the main function to run the program.

Scenario2.py:   Similar to Scenario1.py. Only difference is we are now collecting 3 packages.

Scenario3.py:   Similar to Scenario2.py. Only difference is we are now collecting the 3 packages in a sequential order of Red Green then Blue only