import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
Guide:

Run the function play() to play Tetris. Use '<-' and '->' to move the coming tile to the sides and the down arrow to 
move the tile down. (I have not implemented that the tile moves down with ongoing time). Use the space bar to rotate the
tile and press enter to move the tile as far down as possible (There is still a bug with that, but ¯\_(ツ)_/¯ ).

If the game board is filled up the score appears on the board. Press any key to restart. The high score in the bottom 
right corner gets updated if you break it. Have fun i guess.. ;)
"""


class Tetris:

    def __init__(self):
        self.board = np.zeros((20, 10))
        self.tile = Tile().deploy()
        self.next_tiles = [Tile(), Tile(), Tile(), Tile()]
        self.score = 0
        self.game_over = False
        self.high_score = pickle.load(open("Highscore.pickle", "rb"))

    def __str__(self):
        return f'Placed blocks: {int(self.board.sum())} \n' \
               f'Current tile in game: "{self.tile.type}" \n' \
               f'Upcoming tiles: "{self.next_tiles[0].type}", "{self.next_tiles[1].type}", ' \
               f'"{self.next_tiles[2].type}", "{self.next_tiles[3].type}"'

    def __repr__(self):
        return self.__str__()

    def shift_down(self, down=1, draw_on_board=False):
        for i in range(0, down):
            self.tile.center[0] = self.tile.center[0] + 1  # move tile one block down
            if not self.tile.is_on_board() or self.check_overlap():
                # tile walked out of board --> Try to shift less down and write tile on board
                self.tile.center[0] = self.tile.center[0] - 1  # move tile back up
                self.draw_on_board()
                break
        return self

    def check_overlap(self):
        for pos in self.tile.get_positions():
            if 0 <= pos[0] < 20 and 0 <= pos[1] < 10:
                if self.board[pos[0], pos[1]] == 1:
                    return True
        return False

    def shift_sideways(self, shift=0, draw_on_board=False):
        for i in range(0, abs(shift)):
            self.tile.center[1] = self.tile.center[1] + np.sign(shift)  # move tile one block in required direction
            if not self.tile.is_on_board():
                self.tile.center[1] = self.tile.center[1] - np.sign(shift)  # shift tile back
                break
            if self.check_overlap():
                self.tile.center[1] = self.tile.center[1] - np.sign(shift)  # shift tile back
                self.draw_on_board()
                break
        return self

    def rotate(self):
        self.tile.rotate()
        if not self.tile.is_on_board():
            self.tile.rotate(-1)  # rotate back
            return self
        if self.check_overlap():
            self.tile.rotate(-1)  # rotate back
            return self
        return self

    def show(self, playing=False):

        img = np.zeros((220, 200))
        # draw edge around board:
        img[8:212, [8, 111]] = 1
        img[[8, 211], 8:112] = 1

        # add current tile to board:
        board = self.add_tile_to_board()
        # upscale board
        factor = 10
        big_board = board.repeat(factor, axis=0).repeat(factor, axis=1)
        # insert board:
        img[10:210, 10:110] = big_board

        # insert upcoming:
        start_px_hight = 0
        for upcoming in self.next_tiles:
            upcoming_img = upcoming.show(suppress=True)
            # upscale
            upcoming_img = upcoming_img.repeat(factor, axis=0).repeat(factor, axis=1)
            img[start_px_hight:start_px_hight+upcoming_img.shape[0], 120:120+upcoming_img.shape[1]] = upcoming_img
            start_px_hight = start_px_hight + upcoming_img.shape[0] - 10

        # insert score:
        score = self.get_score_pxs()
        img[170:170+score.shape[0], 130:130+score.shape[1]] = score
        # draw score box:
        img[180:193, [130, 180]] = 1
        img[[180, 192], 130:181] = 1
        score_digits = self.get_digit_pxs()
        img[182:182+score_digits.shape[0], 180-score_digits.shape[1]:180] = score_digits

        # insert highscore:
        high_score_digits = self.get_digit_pxs(high_score=True)
        img[-high_score_digits.shape[0]-2: -2, - high_score_digits.shape[1]-2:-2] = high_score_digits

        if self.game_over:
            # draw score on board
            img[90:120, 10:111] = 0
            bigger_score_digits = score_digits.repeat(2, axis=0).repeat(2, axis=1)
            img[95:95 + bigger_score_digits.shape[0], 100 - bigger_score_digits.shape[1]:100] = bigger_score_digits
            if self.score > self.high_score:
                pickle.dump(self.score, open("Highscore.pickle", "wb"))

        if not playing:
            fig1, ax1 = plt.subplots()
            ax1.clear()
            ax1.imshow(img, cmap='gray')
            plt.show()
        else:
            ax.clear()
            ax.imshow(img, cmap='gray')
            plt.show()

        if self.game_over:
            self.__init__()

    def draw_on_board(self):
        on_board_before = self.board.sum()
        self.board = self.add_tile_to_board()
        if self.board.sum() < on_board_before + 4:
            self.game_over = True
            return
        self.score = self.score + 4
        self.erase_complete_rows()
        self.tile = self.next_tiles.pop(0).deploy()
        self.next_tiles.append(Tile())

    def add_tile_to_board(self):
        positions = self.tile.get_positions()
        board = self.board.copy()
        for pos in positions:
            if pos[0] >= 0:
                board[pos[0], pos[1]] = 1
        return board

    def erase_complete_rows(self):
        incomplete_rows = self.board.sum(axis=1) < 10
        # erase rows:
        self.board = self.board[incomplete_rows, :]
        # add empty rows:
        new_rows = np.zeros((20-incomplete_rows.sum(), 10))
        self.board = np.concatenate([new_rows, self.board])
        # adjust score
        self.score = self.score + 10 * (20 - incomplete_rows.sum())**2

    def deploy_next_tile(self):
        self.tile = self.next_tiles.pop(0).deploy()
        self.next_tiles.append(Tile())

    def get_score_pxs(self):
        return \
            np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                      [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0]])

    def get_digit_pxs(self, high_score=False):
        digits = np.array([
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])
        score_mat = np.zeros((9, 0))
        if high_score:
            score = self.high_score
        else:
            score = self.score
        for digit in str(score):
            score_mat = np.concatenate([score_mat, digits[:, 6*int(digit):6*(int(digit)+1)]], axis=1)
        return score_mat

    def make_nn_move(self, nn_command):
        """ This function makes a move on the tetris game based on the output of the neural network. you might have to
        change this function depending on the network output structure.
        :param self:
        :param nn_command:
        :return:
        """
        [move_l, move_r, stay, rotation] = nn_command
        # firstly rotate:
        if rotation >= 0.5:
            self.rotate()
        # secondly shift left or right
        if move_l > move_r and move_l > stay:
            self.shift_sideways(shift=-1)
        elif move_r > move_l and move_r > stay:
            self.shift_sideways(shift=1)
        # thirdly move down:
        self.shift_down(down=1)


class Tile:

    def __init__(self, type=None):

        if type is None:
            type = np.random.choice(['T', 'I', 'O', 'L1', 'L2', 'S1', 'S2'], p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
        elif type not in ['T', 'L1', 'L2', 'I', 'O', 'S1', 'S2']:
            raise ValueError(f'There is no tile of type {type}.')

        self.type = type
        if type in ['L1', 'L2', 'S1', 'S2', 'T', 'I']:
            self.center = np.array([0, 0])
        if type in ['O']:
            self.center = np.array([0.5, 0.5])
        self.orientation = 0

    def deploy(self):
        self.center = self.center + [0, 5]
        self.orientation = np.random.randint(4)
        if self.type == 'I' and self.orientation == 1:
            self.center = self.center + [1, 0]
        return self

    def __str__(self):
        return f'"{self.type}" at position {self.center} with orientation {self.orientation}'

    def __repr__(self):
        return f'"{self.type}", {self.center}, {self.orientation}'

    def rotate(self, roations=1):
        self.orientation = (self.orientation + roations) % 4

    def get_positions(self):
        r = np.array([[0, -1], [1, 0]])  # 90 deg rotation matrix (counter clockwise)
        if self.type == 'L1':
            positions = np.array([(0, 1), (0, 0), (0, -1), (1, -1)])
        if self.type == 'L2':
            positions = np.array([(0, 1), (0, 0), (0, -1), (-1, -1)])
        if self.type == 'S1':
            positions = np.array([(0, -1), (0, 0), (1, 0), (1, 1)])
        if self.type == 'S2':
            positions = np.array([(0, -1), (0, 0), (-1, 0), (-1, 1)])
        if self.type == 'O':
            positions = np.array([(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)])
        if self.type == 'T':
            positions = np.array([(-1, 0), (0, 0), (1, 0), (0, -1)])
        if self.type == 'I':
            positions = np.array([(0, 1), (0, 0), (0, -1), (0, -2)])

        # rotate:
        for i in range(0, self.orientation):
            positions = positions.dot(r)

        # move to center
        return (positions + self.center).astype(int)

    def show(self, suppress=False):
        positions = self.get_positions()
        # assert that min of x and y coordinate are 0:
        positions[:, 0] = positions[:, 0] - positions[:, 0].min()
        positions[:, 1] = positions[:, 1] - positions[:, 1].min()
        max_x = positions[:, 0].max()
        max_y = positions[:, 1].max()

        # create img, one pixel bigger as tile in each direction
        img = np.zeros((max_x+3, max_y+3))
        # color in tile positions
        for pos in positions:
            img[pos[0]+1, pos[1]+1] = 1
        # show tile
        if not suppress:
            plt.imshow(img, cmap='gray')

        return img

    def is_on_board(self, board_x=20, board_y=10):
        positions = self.get_positions()
        if positions[:, 0].min() < 0 or positions[:, 0].max() >= board_x:
            return False
        elif positions[:, 1].min() < 0 or positions[:, 1].max() >= board_y:
            return False
        else:
            return True


def onclick(event):
    print(event.key)
    if event.key is 'down':
        game.shift_down()
    if event.key is 'left':
        game.shift_sideways(shift=-1)
    if event.key is 'right':
        game.shift_sideways(shift=1)
    if event.key is ' ':
        game.rotate()
    if event.key is 'enter':
        game.shift_down(down=20)
    game.show(playing=True)


def play():
    global fig, ax, game
    fig, ax = plt.subplots()
    game = Tetris()
    game.show(playing=True)

    for i in range(0, 1):
        cid = fig.canvas.mpl_connect('key_press_event', onclick)


def watch_NN_play_tetris(neural_network=None):
    if neural_network is None:
        neural_network = pickle.load(open("Best_NN.pickle", "rb"))
    global fig, ax, game, nn
    fig, ax = plt.subplots()
    game = Tetris()
    game.show(playing=True)
    nn = neural_network

    for i in range(0, 1):
        cid = fig.canvas.mpl_connect('key_press_event', onclick_nn)


def onclick_nn(event):
    input_vector = game.board.flatten()
    nn_command = nn.predict(input_vector)
    game.make_nn_move(nn_command)
    game.show(playing=True)



