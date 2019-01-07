import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

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
        self.tile.center[0] = self.tile.center[0] + down  # move tile one block down
        if not self.tile.is_on_board() or self.check_overlap():
            # tile walked out of board --> Try to shift less down and write tile on board
            self.tile.center[0] = self.tile.center[0] - down  # move tile back up
            return self.shift_down(down=down-1, draw_on_board=True)
        if draw_on_board:
            self.draw_on_board()
        return self

    def check_overlap(self):
        for pos in self.tile.get_positions():
            if 0 <= pos[0] < 20 and 0 <= pos[1] < 10:
                if self.board[pos[0], pos[1]] == 1:
                    return True
        return False

    def shift_sideways(self, shift=0, draw_on_board = False):
        self.tile.center[1] = self.tile.center[1] + shift  # move tile one block down
        if not self.tile.is_on_board():
            self.tile.center[1] = self.tile.center[1] - shift  # shift tile back
            return self.shift_sideways(shift=shift-np.sign(shift))
        if self.check_overlap():
            self.tile.center[1] = self.tile.center[1] - shift  # shift tile back
            return self.shift_sideways(shift=shift - np.sign(shift), draw_on_board=True)
        if draw_on_board:
            self.draw_on_board()
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

    def show(self):

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


class Tile:

    def __init__(self, type=None):

        if type is None:
            type = random.choice(['T', 'L1', 'L2', 'I', 'O', 'S1', 'S2'])
        elif type not in ['T', 'L1', 'L2', 'I', 'O', 'S1', 'S2']:
            raise ValueError(f'There is no tile of type {type}.')

        self.type = type
        self.center = np.array([0, 0])
        self.orientation = 0

    def deploy(self):
        self.center = np.array([0, 5])
        self.orientation = np.random.randint(4)
        return self

    def __str__(self):
        return f'"{self.type}" at position {self.center} with orientation {self.orientation}'

    def __repr__(self):
        return f'"{self.type}", {self.center}, {self.orientation}'

    def rotate(self, roations=1):
        self.orientation = (self.orientation + roations)%4

    def get_positions(self):

        if self.type == 'L1':
            positions = [(-1, 1), (-1, 0), (-1, -1), (0, -1)]
        if self.type == 'L2':
            positions = [(0, 1), (0, 0), (0, -1), (-1, -1)]
        if self.type == 'S1':
            positions = [(-1, 0), (-1, -1), (0, -1), (0, -2)]
        if self.type == 'S2':
            positions = [(0, 0), (0, -1), (-1, -1), (-1, -2)]
        if self.type == 'O':
            positions = [(-1, 0), (-1, -1), (0, -1), (0, 0)]
        if self.type == 'T':
            positions = [(-1, 0), (-1, -1), (0, -1), (-1, -2)]
        if self.type == 'I':
            positions = [(0, 1), (0, 0), (0, -1), (0, -2)]

        positions = self.rotate_positions(positions, self.orientation)
        positions = self.shift_raw_positions(positions, self.center)

        return positions

    def rotate_positions(self, positions, num_of_rot):
        r = np.array([[0, -1], [1, 0]])  # 90 deg rotation matrix (counter clockwise)
        for i in range(0, num_of_rot):
            for pos in range(0, len(positions)):
                positions[pos] = (r.dot(positions[pos] + np.array([0.5, 0.5])) - np.array([0.5, 0.5])).astype(int)
        return positions

    def shift_raw_positions(self, positions, center):
        for pos in range(0, len(positions)):
            positions[pos] = positions[pos] + center
        return positions

    def show(self, suppress=False):
        positions = self.get_positions()
        # assert that min of x and y coordinate are 0:
        min_x = np.inf
        min_y = np.inf
        max_x = - np.inf
        max_y = - np.inf
        for pos in positions:
            if pos[0] < min_x:
                min_x = pos[0]
            if pos[1] < min_y:
                min_y = pos[1]
            if pos[0] > max_x:
                max_x = pos[0]
            if pos[1] > max_y:
                max_y = pos[1]
        for pos in positions:
            pos[0] = pos[0] - min_x
            pos[1] = pos[1] - min_y
        max_x = max_x - min_x
        max_y = max_y - min_y

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
        for pos in positions:
            if pos[1] < 0 or pos[1] >= board_y:
                return False
            if pos[0] >= board_x:
                return False
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
    game.show()

def play():
    global fig, ax, game
    fig, ax = plt.subplots()
    game = Tetris()

    for i in range(0, 1):
        cid = fig.canvas.mpl_connect('key_press_event', onclick)






