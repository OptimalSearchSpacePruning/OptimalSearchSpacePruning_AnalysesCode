import random
import numpy as np
import copy
import math


def convert_ab_board_to_matrix(ab_board):
    board_matrix = []
    dimension = int(math.sqrt(len(ab_board)))
    i = 1
    for row in range(dimension):
        board_matrix.append([])
        for col in range(dimension):
            board_matrix[row].append(ab_board[i])
            i += 1
    return board_matrix


def convert_numeric_board_to_xo(numeric_board):
    board_matrix = np.zeros((len(numeric_board), len(numeric_board)))
    board_matrix = board_matrix.tolist()
    for i in range(len(numeric_board)):
        for j in range(len(numeric_board[i])):
            board_matrix[i][j] = numeric_board[i][j]
            if numeric_board[i][j] == 1:
                board_matrix[i][j] = 'X'
            elif numeric_board[i][j] == 2:
                board_matrix[i][j] = 'O'

    return board_matrix

def convert_position_to_row_col(pos, dimension):
    col = int(((pos - 1) % dimension))
    row = (float(pos)/float(dimension))-1
    row = int(math.ceil(row))
    return [row, col]



def convert_position_to_int(row, col, dimension):
    return row * dimension + col + 1



def get_winning_paths(board_size):
    if board_size == 6:
        file_path = "examples/board_6_4.txt"
    elif board_size == 10:
        file_path = "examples/board_10_5.txt"
    else:
        print('invalid board size, needs to be 6 or 10')
        return

    '''Opens the file, processes the input, and initailizes the Game() object'''
    try:
        input_file = open(file_path ,'r')
    except:
        raise Exception("File not found!!! Make sure you didn't make a spelling error.")
    num_spaces = int(input_file.readline())
    winning_paths = []
    for line in input_file:
        path = map(int, line.split())
        winning_paths.append(list(path))

    return winning_paths


def check_path_clear(board, path, opponent):
    for pos in path:
        row, col = convert_position_to_row_col(pos, len(board))
        if board[row][col] == opponent:
            return False

    return True


def covert_pos_path_to_row_col_path(path, dimension):
    row_col_path = []
    for p in path:
        row_col_path.append(convert_position_to_row_col(p, dimension))
    return row_col_path


def get_open_paths_through_square_method2(row_prev, col_prev, board, player):
    if player == 1:
        opponent = 2
    elif player == 2:
        opponent = 1
    else:
        print('player needs to be either 1 or 2')

    open_paths_data = []

    winning_paths = get_winning_paths(len(board))
    square_pos = convert_position_to_int(row_prev, col_prev, len(board))
    for path in winning_paths:
        if square_pos in path:
            if check_path_clear(board, path, opponent):
                open_paths_data.extend(covert_pos_path_to_row_col_path(path, len(board)))

    return open_paths_data


def get_open_paths_through_square_method2_x_o(row_prev, col_prev, board, player):
    if player == 'X':
        opponent = 'O'
    elif player == 'O':
        opponent = 'X'
    else:
        print('player needs to be either X or O')

    open_paths_data = []

    winning_paths = get_winning_paths(len(board))
    square_pos = convert_position_to_int(row_prev, col_prev, len(board))
    for path in winning_paths:
        if square_pos in path:
            if check_path_clear(board, path, opponent):
                for square in path:
                    sq_pos = convert_position_to_row_col(square, len(board))
                    if sq_pos not in open_paths_data:
                        open_paths_data.append(sq_pos)
                # open_paths_data.extend(covert_pos_path_to_row_col_path(path, len(board)))

    return open_paths_data


def get_open_paths_through_square(row_prev, col_prev, board, player):
    # other_player = 'O'
    if player == 1:
        other_player = 2
    else:
        other_player = 1

    max_length_path = 0
    threshold = -1
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for row in range(len(board)):
        for col in range(len(board)):
            for i in range(streak_size):
                r = row - i
                c = col - i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                empty_squares = []
                path = []
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                    else:
                        empty_squares.append([square_row,square_col])
                    path.append([square_row,square_col])
                    square_row += 1
                    square_col += 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold) & ([row_prev, col_prev] in path):  # add the path if it's not blocked and if there is already at least one X on it
                    open_paths_data.extend(path)

            # check left-down diagonal
            for i in range(streak_size):
                r = row - i
                c = col + i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                empty_squares = []
                path = []
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                    else:
                        empty_squares.append([square_row,square_col])
                    path.append([square_row,square_col])
                    square_row += 1
                    square_col -= 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold) & ([row_prev, col_prev] in path):  # add the path if it's not blocked and if there is already at least one X on it
                    open_paths_data.extend(path)


            # check vertical
            for i in range(streak_size):
                r = row - i
                c = col
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                empty_squares = []
                path = []
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                    else:
                        empty_squares.append([square_row,square_col])

                    path.append([square_row,square_col])
                    square_row += 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold) & ([row_prev, col_prev] in path):  # add the path if it's not blocked and if there is already at least one X on it
                    open_paths_data.extend(path)

            # check horizontal
            for i in range(streak_size):
                r = row
                c = col - i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                empty_squares = []
                path = []
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                    else:
                        empty_squares.append([square_row, square_col])

                    path.append([square_row, square_col])
                    square_col += 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold) & ([row_prev, col_prev] in path):  # add the path if it's not blocked and if there is already at least one X on it
                    open_paths_data.extend(path)

    return remove_duplicates(open_paths_data)



def get_open_paths_through_square_old(row, col, board, player='X'):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    max_length_path = 0
    threshold = -1
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 2) & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.extend(path)

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player==1):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 2:
                open_paths_data.extend(path)
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif path_x_count > threshold:
                open_paths_data.extend(path)

    return remove_duplicates(open_paths_data)



def remove_duplicates(square_list):
    unique_squares = []
    for square in square_list:
        if not check_square_in_list(square, unique_squares):
            unique_squares.append(square)
    return unique_squares


def expand_neighborhood(squares, size, prob=1.0):
    checked = []
    new_neighborhood = []
    for square in squares:
        if str(square) not in checked:
            neighbors = get_neighboring_squares(size, square, 1)
            for neighbor in neighbors:
                if not check_square_in_list(neighbor,new_neighborhood):
                    if random.random() <= prob:
                        new_neighborhood.append(neighbor)
                    # else:
                    #     print 'not adding'
            checked.append(str(square))
            # new_neighborhood.extend(get_neighboring_squares(size, square, 1))
    return remove_duplicates(new_neighborhood)


def check_square_in_list(square, squares_list):
    for i in range(len(squares_list)):
        if (square[0] == squares_list[i][0]) & (square[1] == squares_list[i][1]):
            return True

    return False

def get_neighboring_squares(size, square, neighborhood_size):
    neighbors = []
    row = square[0]
    col = square[1]
    for i in range(-1*neighborhood_size,neighborhood_size+1):
        for j in range(-1*neighborhood_size,neighborhood_size+1):
            if (i != 0) | (j != 0):
                r = row + i
                c = col + j
                if (r < size) & (r >= 0) & (c < size) & (c >= 0):
                    neighbors.append([r,c])
    return neighbors



def rand_max(iterable, key=None):
    """
    A max function that tie breaks randomly instead of first-wins as in
    built-in max().
    :param iterable: The container to take the max from
    :param key: A function to compute tha max from. E.g.:
      >>> rand_max([-2, 1], key=lambda x:x**2
      -2
      If key is None the identity is used.
    :return: The entry of the iterable which has the maximum value. Tie
    breaks are random.
    """
    if key is None:
        key = lambda x: x

    max_v = -np.inf
    max_l = []

    for item, value in zip(iterable, [key(i) for i in iterable]):
        # print item
        # print value
        if value == max_v:
            max_l.append(item)
        elif value > max_v:
            max_l = [item]
            max_v = value

    return random.choice(max_l)


def dist_in_matrix(index_i, index_j, dim):
    r1,c1 = convert_position_to_row_col(index_i, dim)
    r2,c2 = convert_position_to_row_col(index_j, dim)
    return max(abs(r1-r2),abs(c1-c2))


def generate_matrix_dist_metric(dim, norm=True):
    distances = np.zeros((dim*dim, dim*dim))
    for i in range(np.size(distances, 1)):
        for j in range(np.size(distances, 1)):
            # print(i-1)
            # print(j-1)
            distances[i][j] = dist_in_matrix(i+1,j+1,dim)
    if norm:
        distances = distances/distances.sum()
        # first_move_matrix/first_move_matrix.sum()
    return distances

