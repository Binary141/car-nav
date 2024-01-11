import numpy as np
import copy
import random
import math

class CarNavigationState:
    def __init__(self, row_count=1, column_count=5, numRotated=2):
        self._paths = np.full((row_count, column_count), 0, dtype=np.int8) # np.zeros(size, dtype=np.int8)
        self._size = row_count * column_count
        self._rows = row_count
        self._cols = column_count

        # Make the goal in the lowest right hand corner
        self._goal_row = row_count - 1
        self._goal_col = column_count - 1

        self._possibleOptions = [0, 1, 2, 3]
        self._possibleOptionsConverted = ["-", "\\", "|", "/"]
        self._agent_row = 0
        self._agent_col = 0

        if numRotated > row_count * column_count:
            self._numRotated = (row_count * column_count)
            self.rotateAll()
            return

        self._numRotated = numRotated
        self.rotateNumTimes(numRotated)
        self._action = None # int - action took to get to this node
        self._estimated_cost = 0
        self._cost = 0
        self._parent = None
        return

    # For use in the prio queue
    def __lt__(self, other):
        return self._estimated_cost < other._estimated_cost

    @property
    def size(self):
        return self._size

    def rotateAll(self):
        for row, pathList in enumerate(self._paths):
            for col in range(len(pathList)):
                self._paths[row][col] = self._possibleOptions[random.randrange(0,len(self._possibleOptions))]

        if self._rows == 1:
            # make the first one a '-' since we won't be able to do anything otherwise
            self._paths[0][0] = 0
        else:
            self._paths[0][0] = self._possibleOptions[:-1][random.randrange(0,len(self._possibleOptions)-1)]

    def rotateNumTimes(self, count):
        for i in range(count):
            rowIndx = random.randrange(0, self._rows)
            colIndx = random.randrange(0, self._cols)
            while self._paths[rowIndx][colIndx] != 0:
                # Don't rotate pieces that were already rotated
                rowIndx = random.randrange(0, self._rows)
                colIndx = random.randrange(0, self._cols)
            # Skip the first option so we can't randomly stay at the same place
            self._paths[rowIndx][colIndx] =\
                self._possibleOptions[1:][random.randrange(0,\
                   len(self._possibleOptions)-1)]

    def randomize(self, seed=None):
        self.rotateAll()

    def rotatePiece(self, row, col, action):
        currPath = self._paths[row][col]
        if action == "right":
            # Rotating a '-'
            if currPath == 0:
                # Make it a '\'
                self._paths[row][col] = 1
            # Rotating a '\'
            if currPath == 1:
                # Make it a '|'
                self._paths[row][col] = 2
            # Rotating a '|'
            if currPath == 2:
                # Make it a '/'
                self._paths[row][col] = 3
            # Rotating a '/'
            if currPath == 3:
                # Make it a '-'
                self._paths[row][col] = 0
            return

        if action == "left":
            # Rotating a '-'
            if currPath == 0:
                # Make it a '/'
                self._paths[row][col] = 3
            # Rotating a '\'
            if currPath == 1:
                # Make it a '-'
                self._paths[row][col] = 0
            # Rotating a '|'
            if currPath == 2:
                # Make it a '\'
                self._paths[row][col] = 1
            # Rotating a '/'
            if currPath == 3:
                # Make it a '|'
                self._paths[row][col] = 2
            return

        raise Exception(f"Action {action} not recognized!!")

    @property
    def observation(self):
        return self._paths

    @observation.setter
    def observation(self, value):
        self._paths = value
        self._size = value.shape[0]
        return

    def path(self, row, col):
        return self._paths[row][col]

    def getAgentLocation(self):
        return (self._agent_row, self._agent_col)

    def getGoalLocation(self):
        return (self._goal_row, self._goal_col)

    def setAgentLocation(self, location):
        self._agent_row, self._agent_col = location

    def getCurrentPath(self):
        return self._paths[self._agent_row][self._agent_col]

    def getRowCount(self):
        return self._rows

    def getColumnCount(self):
        return self._cols

    def getPath(self, row, col):
        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            return ""
        return self._paths[row][col]

    def __str__(self):
        # s = f"agent at row, col {self._agent_row}, {self._agent_col}\n"
        # s += f"goal at row, col {self._goal_row}, {self._goal_col}\n\n"
        s = ""

        existingAgentPath = self._paths[self._agent_row][self._agent_col]
        existingGoalPath = self._paths[self._goal_row][self._goal_col]

        self._paths[self._goal_row][self._goal_col] = 50
        self._paths[self._agent_row][self._agent_col] = 100

        for pathList in self._paths:
            s += " "
            for path in pathList:
                if path == 50:
                    s += " G "
                elif path == 100:
                    s += " A "
                else:
                    s += f" {self._possibleOptionsConverted[path]} "
            s += "\n"
        self._paths[self._agent_row][self._agent_col] = existingAgentPath
        self._paths[self._goal_row][self._goal_col] = existingGoalPath
        return s

class CarNavigationModel:

    def ACTIONS(state):
        """
        --- Movement ---
        mr - right
        ml - left
        mu - up
        md - down
        mur - up right
        mul - up left
        mdr - down right
        mdl - down left

        --- Rotation ---
        rrr - right piece to the right
        rrl - right piece to the left
        rlr - left piece right
        rll - left piece left
        rtr - top piece right
        rtl - top piece left
        rbr - bottom piece right
        rbl - bottom piece left
        rtrr - top right piece right
        rtrl - top right piece left
        rblr - bottom left piece right
        rbll - bottom left piece left
        rtlr - top left piece right
        rtll - top left piece left
        rbrr - bottom right piece right
        rbrl - bottom right piece left
        """
        currPath = state.getCurrentPath()
        row, col = state.getAgentLocation()
        actions = []

        # If we are looking at a '-'
        if currPath == 0:
            if col - 1 >= 0:
                actions.append("ml")
                actions.append("rlr")
                actions.append("rll")
            if col + 1 < state.getColumnCount():
                actions.append("mr")
                actions.append("rrr")
                actions.append("rrl")

        # If we are looking at a '|'
        if currPath == 2:
            if row - 1 >= 0:
                actions.append("mu")
                actions.append("rtr")
                actions.append("rtl")
            if row + 1 < state.getRowCount():
                actions.append("md")
                actions.append("rbr")
                actions.append("rbl")

        # If we are looking at a '/'
        if currPath == 3:
            if (row - 1 >= 0) and (col + 1 < state.getColumnCount()):
                actions.append("mur")
                actions.append("rtrr")
                actions.append("rtrl")
            if (row + 1 < state.getRowCount()) and (col - 1 >= 0):
                actions.append("mdl")
                actions.append("rblr")
                actions.append("rbll")

        # If we are looking at a '\'
        if currPath == 1:
            if (row - 1 >= 0) and (col - 1 >= 0):
                actions.append("mul")
                actions.append("rtlr")
                actions.append("rtll")
            if (row + 1 < state.getRowCount()) and (col + 1 < state.getColumnCount()):
                actions.append("mdr")
                actions.append("rbrr")
                actions.append("rbrl")

        return actions

    def RESULT(state, action):
        if action is None or action[0] not in ["m", "r"]:
            raise Exception(f"Action {action} not recognized in result!")

        state1 = copy.deepcopy(state)

        row, col = state1.getAgentLocation()

        if action == "mr":
            state1.setAgentLocation((row, col+1))
        elif action == "ml":
            state1.setAgentLocation((row, col-1))

        elif action == "mu":
            state1.setAgentLocation((row-1, col))
        elif action == "md":
            state1.setAgentLocation((row+1, col))

        elif action == "mur":
            state1.setAgentLocation((row-1, col+1))
        elif action == "mul":
            state1.setAgentLocation((row-1, col-1))

        elif action == "mdr":
            state1.setAgentLocation((row+1, col+1))
        elif action == "mdl":
            state1.setAgentLocation((row+1, col-1))

        elif action == "rrr":
            state1.rotatePiece(row, col+1, "right")
        elif action == "rrl":
            state1.rotatePiece(row, col+1, "left")

        elif action == "rlr":
            state1.rotatePiece(row, col-1, "right")
        elif action == "rll":
            state1.rotatePiece(row, col-1, "left")

        elif action == "rtr":
            state1.rotatePiece(row-1, col, "right")
        elif action == "rtl":
            state1.rotatePiece(row-1, col, "left")

        elif action == "rbr":
            state1.rotatePiece(row+1, col, "right")
        elif action == "rbl":
            state1.rotatePiece(row+1, col, "left")

        elif action == "rtrr":
            state1.rotatePiece(row-1, col+1, "right")
        elif action == "rtrl":
            state1.rotatePiece(row-1, col+1, "left")

        elif action == "rblr":
            state1.rotatePiece(row+1, col-1, "right")
        elif action == "rbll":
            state1.rotatePiece(row+1, col-1, "left")

        elif action == "rtlr":
            state1.rotatePiece(row-1, col-1, "right")
        elif action == "rtll":
            state1.rotatePiece(row-1, col-1, "left")

        elif action == "rbrr":
            state1.rotatePiece(row+1, col+1, "right")
        elif action == "rbrl":
            state1.rotatePiece(row+1, col+1, "left")

        else:
            raise Exception(f"Movement {action} not recognized!")
        return state1

    def GOAL_TEST(state):
        return state.getAgentLocation() == state.getGoalLocation()

    def STEP_COST(state, action, state1):
        if action in ["mur", "mul", "mdr", "mdl"]:
            # These move diagonally, equivalent to a move up/down and then right/left
            cost = 2
        else:
            cost = 1
        return cost

    def HEURISTIC(state):
        # Gets the distance to the goal from the agents position
        agent_row, agent_col = state.getAgentLocation()
        goal_row, goal_col = state.getGoalLocation()

        return math.sqrt((agent_row - goal_row) ** 2 + (agent_col - goal_col) ** 2)

if __name__ == "__main__":
    s = CarNavigationState(row_count=5, column_count=5, numRotated=50)
    print("s is:")
    print(s)
    print(s.rotatePiece(1, 1, "right"))
    print(s)
    print(s.rotatePiece(1, 1, "right"))
    print(s)
    print(s.rotatePiece(1, 1, "right"))
    print(s)
    print(s.rotatePiece(1, 1, "right"))
    print(s)

    print(CarNavigationModel.ACTIONS(s))
    print(CarNavigationModel.HEURISTIC(s))
