import numpy as np
import matplotlib.pyplot as plt
import random


class Tetris:

    def __init__(self):
        self.board = np.zeros((20, 10))
        self.tile = Tile().deploy()
        self.next_tiles = [Tile(), Tile(), Tile(), Tile()]

    def __str__(self):
        return f'Placed blocks: {int(self.board.sum())} \n' \
               f'Current tile in game: "{self.tile.type}" \n' \
               f'Upcoming tiles: "{self.next_tiles[0].type}", "{self.next_tiles[1].type}", ' \
               f'"{self.next_tiles[2].type}", "{self.next_tiles[3].type}"'

    def __repr__(self):
        return self.__str__()

    def show(self):

        img = np.zeros((220, 200))
        # draw edge around board:
        img[8:212, [8, 111]] = 1
        img[[8, 211], 8:112] = 1

        # add current tile to board:
        self.add_tile_to_board()
        # upscale board
        factor = 10
        big_board = self.board.repeat(factor, axis=0).repeat(factor, axis=1)
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

        plt.imshow(img, cmap='gray')

    def add_tile_to_board(self):
        positions = self.tile.get_positions()
        for pos in positions:
            if pos[0] >= 0:
                self.board[pos[0], pos[1]] = 1

    def deploy_next_tile(self):
        self.tile = self.next_tiles.pop(0).deploy()
        self.next_tiles.append(Tile())


class Tile:

    def __init__(self, type=None):

        if type is None:
            type = random.choice(['T', 'L1', 'L2', 'I', 'O', 'S1', 'S2'])
        elif type not in ['T', 'L1', 'L2', 'I', 'O', 'S1', 'S2']:
            raise ValueError(f'There is no tile of type {type}.')

        self.type = type
        self.center = (0, 0)
        self.orientation = 0

    def deploy(self):
        self.center = (0, 5)
        self.orientation = np.random.randint(4)
        return self

    def __str__(self):
        return f'"{self.type}" at position {self.center} with orientation {self.orientation}'

    def __repr__(self):
        return f'"{self.type}", {self.center}, {self.orientation}'

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

        positions = self.rotate(positions, self.orientation)
        positions = self.shift(positions, self.center)

        return positions

    def rotate(self, positions, num_of_rot):
        r = np.array([[0, -1], [1, 0]])  # 90 deg rotation matrix (counter clockwise)
        for i in range(0, num_of_rot):
            for pos in range(0, len(positions)):
                positions[pos] = r.dot(positions[pos])
        return positions

    def shift(self, positions, center):
        for pos in range(0, len(positions)):
            positions[pos] = positions[pos] + np.array(center)
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





