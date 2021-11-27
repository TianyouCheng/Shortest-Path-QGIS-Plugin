from math import sqrt
import queue
import collections

sqrt2 = sqrt(2)


def dijkstra(start_tuple, end_tuples, block, find_nearest, feedback=None):

    class Grid:
        def __init__(self, matrix):
            self.map = matrix
            self.h = len(matrix)
            self.w = len(matrix[0])
            self.manhattan_boundry = None
            self.curr_boundry = None

        def _in_bounds(self, id):
            x, y = id
            return 0 <= x < self.h and 0 <= y < self.w

        def _passable(self, id):
            x, y = id
            return self.map[x][y] is not None

        def is_valid(self, id):
            return self._in_bounds(id) and self._passable(id)

        def neighbors(self, id):
            x, y = id
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1),
                       (x + 1, y - 1), (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1)]
            results = list(filter(self.is_valid, results))
            return results

        @staticmethod
        def manhattan_distance(id1, id2):
            x1, y1 = id1
            x2, y2 = id2
            return abs(x1 - x2) + abs(y1 - y2)

        @staticmethod
        def min_manhattan(curr_node, end_nodes):
            return min(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def max_manhattan(curr_node, end_nodes):
            return max(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def all_manhattan(curr_node, end_nodes):
            return {end_node: Grid.manhattan_distance(curr_node, end_node) for end_node in end_nodes}

        def simple_cost(self, cur, nex):
            cx, cy = cur
            nx, ny = nex
            currV = self.map[cx][cy]
            offsetV = self.map[nx][ny]
            if cx == nx or cy == ny:
                return (currV + offsetV) / 2
            else:
                return sqrt2 * (currV + offsetV) / 2

    result = []
    grid = Grid(block)

    end_dict = collections.defaultdict(list)
    for end_tuple in end_tuples:
        end_dict[end_tuple[0]].append(end_tuple)
    end_row_cols = set(end_dict.keys())
    end_row_col_list = list(end_row_cols)
    start_row_col = start_tuple[0]


    frontier = queue.PriorityQueue()
    frontier.put((0, start_row_col))
    came_from = {}
    cost_so_far = {}
    decided = set()

    if not grid.is_valid(start_row_col):
        return result

    # init progress
    index = 0
    distance_dic = grid.all_manhattan(start_row_col, end_row_cols)
    if find_nearest:
        total_manhattan = min(distance_dic.values())
    else:
        total_manhattan = sum(distance_dic.values())

    total_manhattan = total_manhattan + 1
    bound = total_manhattan
    # feedback.setProgress(1 + 100 * (1 - bound / total_manhattan))

    came_from[start_row_col] = None
    cost_so_far[start_row_col] = 0

    while not frontier.empty():
        # print(result[-1] if len(result)>0 else '')
        _, current_node = frontier.get()
        if current_node in decided:
            continue
        decided.add(current_node)

        # update the progress bar
        if feedback:
            if feedback.isCanceled():
                return None

            index = (index + 1) % len(end_row_col_list)
            target_node = end_row_col_list[index]
            new_manhattan = grid.manhattan_distance(current_node, target_node)
            if new_manhattan < distance_dic[target_node]:
                if find_nearest:
                    curr_bound = new_manhattan
                else:
                    curr_bound = bound - (distance_dic[target_node] - new_manhattan)

                distance_dic[target_node] = new_manhattan

                if curr_bound < bound:
                    bound = curr_bound
                    feedback.setProgress(1 + 100 * (1 - bound / total_manhattan)*(1 - bound / total_manhattan))

        # reacn destination
        if current_node in end_row_cols:
            path = []
            costs = []
            traverse_node = current_node
            while traverse_node is not None:
                path.append(traverse_node)
                costs.append(cost_so_far[traverse_node])
                traverse_node = came_from[traverse_node]
            
            # start point and end point overlaps
            if len(path) == 1:
                path.append(start_row_col)
                costs.append(0.0)
            path.reverse()
            costs.reverse()
            result.append((path, costs, end_dict[current_node]))

            end_row_cols.remove(current_node)
            end_row_col_list.remove(current_node)
            if len(end_row_cols) == 0 or find_nearest:
                break

        # relax distance
        for nex in grid.neighbors(current_node):
            new_cost = cost_so_far[current_node] + grid.simple_cost(current_node, nex)
            if nex not in cost_so_far or new_cost < cost_so_far[nex]:
                cost_so_far[nex] = new_cost
                frontier.put((new_cost, nex))
                came_from[nex] = current_node

    return result