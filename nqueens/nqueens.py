import random
import math
import time

# Implement a solver that returns a list of queen's locations
#  - Make sure the list is the right length, and uses the numbers from 0 .. BOARD_SIZE-1

def print_board(board):
    # Prints a formated version of the board that supports and size list
    print(end='[')
    length = len(board) - 1
    for i in range(length):
        if i != length:
            print(i,end=', ')
    print(board[-1],end=']')

def solve(n):   # O(sqrt(n)  * n) or O(maxSteps * n)
    # Used to re-call the min_conflicts function until a solution is found                                                   
    maxSteps = 2 * ((-0.000006 * n * n) + (0.85 * n) + 138)
    answer = min_conflicts(n, maxSteps)
    while not answer:
        answer = min_conflicts(n, maxSteps)

    # print_board(answer)
    return answer

def is_valid(conflictsPos, conflictsNeg, conflictsRow):
    # Checks validity of a board domain
    for key in conflictsPos:
        if conflictsPos[key] > 1:
            return False
    for key in conflictsNeg:
        if conflictsNeg[key] > 1:
            return False
    for key in conflictsRow:
        if conflictsRow[key] > 1:
            return False

    return True

def set_board(n):
    # Set board using basic heuristic
    if n < 4:
        return False

    board = []
    for i in range(2,n+1,2):
        board.append(i)
    for i in range(1,n+1,2):
        board.append(i)

    return board

# def set_board(n):
#     # Set board such that no queen conflicts with a neighbour and all have unique rows
#     if n < 4:
#         return False

#     initialList = [i for i in range(1, n + 1)]
#     random.shuffle(initialList)
#     board = []
#     board.append(initialList.pop())

#     while len(initialList) > 2:
#         i = -1
#         while initialList[i] == board[-1] + 1 or initialList[i] == board[-1] - 1:
#             i -= 1
#         board.append(initialList[i])
#         del initialList[i]
            
#     board += initialList
#     return board

def min_conflicts(n, max_steps):
    # Move queens using an algorithm that approaches a solution

    # The following can be swapped out to use different initial board set methods
    board = random.sample(range(1, n+1), n)
    # board = set_board(n)

    if not board:
        return False

    coveredPos = {}
    coveredNeg = {}
    coveredRow = {}
    conflictsPos = {}
    conflictsNeg = {}
    conflictsRow = {}
    for column in range(n):
        row = board[column]
        pos = row - column
        neg = row + column
        if pos in coveredPos:
            coveredPos[pos].append(column)
            conflictsPos[pos] += 1
        else:
            coveredPos[pos] = [column]
            conflictsPos[pos] = 1
        if neg in coveredNeg:
            coveredNeg[neg].append(column)
            conflictsNeg[neg] += 1
        else:
            coveredNeg[neg] = [column]
            conflictsNeg[neg] = 1
        if row in coveredRow:
            coveredRow[row].append(column)
            conflictsRow[row] += 1
        else:
            coveredRow[row] = [column]
            conflictsRow[row] = 1

    if is_valid(conflictsPos, conflictsNeg, conflictsRow):
        return board

    k = 1
    conflicts = {}
    keys = []
    for column in range(n):
        row = board[column]
        pos = row - column
        neg = row + column
        numConflicts = -3
        if pos in conflictsPos:
            numConflicts += conflictsPos[pos]
        if neg in conflictsNeg:
            numConflicts += conflictsNeg[neg]
        if row in conflictsRow:
            numConflicts += conflictsRow[row]
        if numConflicts > 0:
            conflicts[column] = numConflicts
            keys.append(column)
    conflictsSize = len(keys)

    while (k < max_steps):
        queen = random.choice(keys)

        minConflicts = n
        rows = []
        values = []
        for row in range(1, n + 1):
            pos = row - queen
            neg = row + queen
            if row == board[queen]:
                numConflicts = -3
                if pos in conflictsPos:
                    numConflicts += conflictsPos[pos]
                if neg in conflictsNeg:
                    numConflicts += conflictsNeg[neg]
                if row in conflictsRow:
                    numConflicts += conflictsRow[row]
            else:
                numConflicts = 0
                if pos in conflictsPos:
                    numConflicts += conflictsPos[pos]
                if neg in conflictsNeg:
                    numConflicts += conflictsNeg[neg]
                if row in conflictsRow:
                    numConflicts += conflictsRow[row]
            values.append(numConflicts)
            rows.append(row)
            if numConflicts < minConflicts:
                minConflicts = numConflicts
                

        choose = []
        for i in range(n):
            if values[i] == minConflicts:
                choose.append(rows[i])
        minRow = random.choice(choose)

        if minConflicts == 0:
            del conflicts[queen]
            keys.remove(queen)
            conflictsSize -= 1
        else:
            conflicts[queen] = minConflicts

        row = board[queen]
        pos = row - queen
        neg = row + queen
        
        coveredPos[pos].remove(queen)
        coveredNeg[neg].remove(queen)
        coveredRow[row].remove(queen)
        conflictsPos[pos] -= 1
        conflictsNeg[neg] -= 1
        conflictsRow[row] -= 1

        row = minRow
        pos = row - queen
        neg = row + queen
        if pos in coveredPos:
            for i in coveredPos[pos]:
                if not i in conflicts:
                    conflicts[i] = 1
                    keys.append(i)
                    conflictsSize += 1
            coveredPos[pos].append(queen)
            conflictsPos[pos] += 1
        else:
            coveredPos[pos] = [queen]
            conflictsPos[pos] = 1
        if neg in coveredNeg:
            for i in coveredNeg[neg]:
                if not i in conflicts:
                    conflicts[i] = 1
                    keys.append(i)
                    conflictsSize += 1
            coveredNeg[neg].append(queen)
            conflictsNeg[neg] += 1
        else:
            coveredNeg[neg] = [queen]
            conflictsNeg[neg] = 1
        if row in coveredRow:
            for i in coveredRow[row]:
                if not i in conflicts:
                    conflicts[i] = 1
                    keys.append(i)
                    conflictsSize += 1
            coveredRow[row].append(queen)
            conflictsRow[row] += 1
        else:
            coveredRow[row] = [queen]
            conflictsRow[row] = 1

        board[queen] = minRow 
        
        if minConflicts == 0 and is_valid(conflictsPos, conflictsNeg, conflictsRow):
            return board
        
        k += 1

    return False