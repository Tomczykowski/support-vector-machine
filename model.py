import pickle
import numpy as np


class SVM:
    def __init__(self, path, learning_rate=0.001, param_lambda=0.01, param_iters=1000):
        self.learning_rate = learning_rate
        self.param_lambda = param_lambda
        self.param_iters = param_iters
        self.path = path
        self.block_size = self.__read_block_size_from_file()
        self.bound_x = self.__read_bounds_from_file()[0]
        self.bound_y = self.__read_bounds_from_file()[1]
        self.X = np.ndarray(shape=(3000, 10), dtype=int)
        self.y = np.ndarray(shape=(3000, 1), dtype=int)
        self.w = np.ndarray(shape=(4, 10), dtype=float)
        self.b = np.ndarray(shape=(4, 1), dtype=float)

    def read_data_from_pickle(self):
        with open(self.path, 'rb') as f:
            data_file = pickle.load(f)
        return data_file

    def __read_block_size_from_file(self):
        data_file = self.read_data_from_pickle()
        return data_file['block_size']

    def __read_bounds_from_file(self):
        data_file = self.read_data_from_pickle()
        return data_file['bounds']

    def fit_direction(self, y, direction):
        n_samples, n_attributes = self.X[:2400].shape
        w = np.zeros(n_attributes)
        b = 0
        y = y[:2400]
        for _ in range(self.param_iters):
            for idx, x_i in enumerate(self.X[:2400]):
                condition = (y[idx] * (np.dot(x_i, w) - b) >= 1)
                if condition:
                    w -= self.learning_rate * (2 * self.param_lambda * w)
                else:
                    w -= self.learning_rate * (2 * self.param_lambda * w - np.dot(x_i, y[idx]))
                    b -= self.learning_rate * y[idx]
        for idx in range(n_attributes):
            self.w.itemset((direction, idx), w[idx])
        self.b.itemset((direction, 0), b)

    def fit(self):
        self.fit_direction(self.change_y_values_for_one_vs_rest_up(), 0)
        self.fit_direction(self.change_y_values_for_one_vs_rest_right(), 1)
        self.fit_direction(self.change_y_values_for_one_vs_rest_down(), 2)
        self.fit_direction(self.change_y_values_for_one_vs_rest_left(), 3)

    def make_attributes(self):
        data_file = self.read_data_from_pickle()
        i = 1
        length_of_snake = 0
        for idx, move in enumerate(data_file["data"]):
            if length_of_snake > len(move[0]['snake_body']):
                i += 1
            if idx - i >= 3000:
                break
            head_x = move[0]['snake_body'][-1][0]
            head_y = move[0]['snake_body'][-1][1]
            course = move[0]['snake_direction'].value
            distance_to_obstacle_up = self.distance_from_obstacle_up(head_x, head_y, move[0]['snake_body'])
            distance_to_obstacle_right = self.distance_from_obstacle_right(head_x, head_y, move[0]['snake_body'])
            distance_to_obstacle_down = self.distance_from_obstacle_down(head_x, head_y, move[0]['snake_body'])
            distance_to_obstacle_left = self.distance_from_obstacle_left(head_x, head_y, move[0]['snake_body'])
            apple_x = move[0]['food'][0]
            apple_y = move[0]['food'][1]
            length_of_snake = len(move[0]['snake_body'])
            y_value = move[1].value
            self.X.itemset((idx - i, 0), head_x // self.block_size + 1)
            self.X.itemset((idx - i, 1), head_y // self.block_size + 1)
            self.X.itemset((idx - i, 2), course)
            self.X.itemset((idx - i, 3), distance_to_obstacle_up)
            self.X.itemset((idx - i, 4), distance_to_obstacle_right)
            self.X.itemset((idx - i, 5), distance_to_obstacle_down)
            self.X.itemset((idx - i, 6), distance_to_obstacle_left)
            self.X.itemset((idx - i, 7), apple_x // self.block_size + 1)
            self.X.itemset((idx - i, 8), apple_y // self.block_size + 1)
            self.X.itemset((idx - i, 9), length_of_snake)
            self.y.itemset(idx - i, y_value)

    def distance_from_obstacle_up(self, x, y, snake_body):
        distance_to_obstacle = y // self.block_size + 1
        for body in snake_body:
            if x == body[0] and y > body[1]:
                distance = (y - body[1]) // self.block_size
                if distance < distance_to_obstacle:
                    distance_to_obstacle = distance
        return distance_to_obstacle

    def distance_from_obstacle_right(self, x, y, snake_body):
        distance_to_obstacle = (self.bound_x - x) // self.block_size
        for body in snake_body:
            if y == body[1] and x < body[0]:
                distance = (body[0] - x) // self.block_size
                if distance < distance_to_obstacle:
                    distance_to_obstacle = distance
        return distance_to_obstacle

    def distance_from_obstacle_down(self, x, y, snake_body):
        distance_to_obstacle = (self.bound_y - y) // self.block_size
        for body in snake_body:
            if x == body[0] and y < body[1]:
                distance = (body[1] - y) // self.block_size
                if distance < distance_to_obstacle:
                    distance_to_obstacle = distance
        return distance_to_obstacle

    def distance_from_obstacle_left(self, x, y, snake_body):
        distance_to_obstacle = x // self.block_size + 1
        for body in snake_body:
            if y == body[1] and x > body[0]:
                distance = (x - body[0]) // self.block_size
                if distance < distance_to_obstacle:
                    distance_to_obstacle = distance
        return distance_to_obstacle

    def change_y_values_for_one_vs_rest_up(self):
        y_values_for_up = []
        for i in self.y:
            y_values_for_up.append(-1) if i == 0 else y_values_for_up.append(1)
        return y_values_for_up

    def change_y_values_for_one_vs_rest_right(self):
        y_values_for_right = []
        for i in self.y:
            y_values_for_right.append(-1) if i == 1 else y_values_for_right.append(1)
        return y_values_for_right

    def change_y_values_for_one_vs_rest_down(self):
        y_values_for_down = []
        for i in self.y:
            y_values_for_down.append(-1) if i == 2 else y_values_for_down.append(1)
        return y_values_for_down

    def change_y_values_for_one_vs_rest_left(self):
        y_values_for_left = []
        for i in self.y:
            y_values_for_left.append(-1) if i == 3 else y_values_for_left.append(1)
        return y_values_for_left

    def game_state_to_data_sample(self, attributes):
        predict_up = np.dot(attributes, self.w[0]) - self.b[0]
        predict_right = np.dot(attributes, self.w[1]) - self.b[1]
        predict_down = np.dot(attributes, self.w[2]) - self.b[2]
        predict_left = np.dot(attributes, self.w[3]) - self.b[3]
        direction = min(predict_up, predict_right, predict_down, predict_left)
        if direction == predict_up:
            return 0
        elif direction == predict_right:
            return 1
        elif direction == predict_down:
            return 2
        elif direction == predict_left:
            return 3

    def make_attributes_from_game_state(self, move):
        attributes = np.ndarray(shape=(1, 10))
        head_x = move['snake_body'][-1][0]
        head_y = move['snake_body'][-1][1]
        course = move['snake_direction'].value
        distance_to_obstacle_up = self.distance_from_obstacle_up(head_x, head_y, move['snake_body'])
        distance_to_obstacle_right = self.distance_from_obstacle_right(head_x, head_y, move['snake_body'])
        distance_to_obstacle_down = self.distance_from_obstacle_down(head_x, head_y, move['snake_body'])
        distance_to_obstacle_left = self.distance_from_obstacle_left(head_x, head_y, move['snake_body'])
        apple_x = move['food'][0]
        apple_y = move['food'][1]
        length_of_snake = len(move['snake_body'])
        attributes.itemset((0, 0), head_x // self.block_size + 1)
        attributes.itemset((0, 1), head_y // self.block_size + 1)
        attributes.itemset((0, 2), course)
        attributes.itemset((0, 3), distance_to_obstacle_up)
        attributes.itemset((0, 4), distance_to_obstacle_right)
        attributes.itemset((0, 5), distance_to_obstacle_down)
        attributes.itemset((0, 6), distance_to_obstacle_left)
        attributes.itemset((0, 7), apple_x // self.block_size + 1)
        attributes.itemset((0, 8), apple_y // self.block_size + 1)
        attributes.itemset((0, 9), length_of_snake)
        return attributes

    def comparison(self):
        result = 0
        y = self.y[2400:]
        for idx, x_i in enumerate(self.X[2400:]):
            if self.game_state_to_data_sample(x_i) == y[idx]:
                result += 1
        return result/600


if __name__ == "__main__":
    svm_player = SVM("data/2022.11.29.22.31.51.pickle")
    svm_player.make_attributes()

    svm_player.fit()
    print(svm_player.w)
    print(svm_player.b)
    # print(svm_player.X)
    print(svm_player.comparison())
