# 使用A_star算法计算最短路径

__all__ = ('a_star',)

import numpy as np
import math

SQRT2 = math.sqrt(2)
SQRT_HALF = math.sqrt(0.5)


def a_star(raster: np.ndarray, start_p, end_p, walk_type):
    '''
    使用a*算法（a_star）算法计算最短路径。
    到终点的估计距离 = 该点到终点经过的格子数（queen步为对角距离、rock步为曼哈顿距离） * raster平均成本

    :param raster: 成本栅格，numpy的矩阵格式
    :param start_p: 起点的点坐标(x, y)，已经转化为行列编号：x是第x列（从0开始），y是"从下往上"第y行（从0开始）
    :param end_p: 终点的点坐标(x, y)，同上
    :param walk_type: str, 表示是`queen邻近`还是`rock邻近`
    :returns: (最低成本距离, 最低成本对应的路径)，路径以numpy矩阵栅格形式返回
    '''
    mean_cost = np.nanmean(raster)
    is_queen = walk_type[:5].lower() == 'queen'
    est_dist_func = queen_distance if is_queen else manhattan_distance
    # 开放列表：格式为{(x, y): [到起点的最小成本, 预估到终点成本, 回溯方向]}
    # 回溯方向是0~7的整数: 0表示向东，1表示东北，每+1表示逆时针增大45°
    open_list = {tuple(start_p): [0, 0, None]}
    close_list = {}   # 封闭列表

    # 循环，计算最小成本，和每个格子的回溯路径
    while True:
        now_p, value = min(open_list.items(), key=lambda kv: kv[1][0] + kv[1][1])
        close_list[now_p] = (value[0], value[2])
        if now_p == end_p:
            break
        # nearby字典的内容为{点坐标: 回溯方向}
        nearby = {(now_p[0] - 1, now_p[1]): 0, (now_p[0], now_p[1] + 1): 2,
                  (now_p[0] + 1, now_p[1]): 4, (now_p[0], now_p[1] - 1): 6}
        if is_queen:
            nearby.update({(now_p[0] - 1, now_p[1] + 1): 1, (now_p[0] + 1, now_p[1] + 1): 3,
                           (now_p[0] + 1, now_p[1] - 1): 5, (now_p[0] - 1, now_p[1] - 1): 7})
        for k, v in nearby:
            if not is_valid(k, raster, close_list):
                continue
            min_dist = value[0] + (raster[now_p[1], now_p[0]] + raster[k[1], k[0]]) * \
                       (0.5 if v % 2 == 0 else SQRT_HALF)
            if k not in open_list:
                est_dist = est_dist_func(k, end_p, mean_cost)
                open_list[k] = [min_dist, est_dist, v]
            elif open_list[k][0] > min_dist:
                open_list[k][0] = min_dist
                open_list[k][2] = v

    # 从close_list计算回溯路径
    route = np.zeros(raster.shape, np.bool_)
    now_p = end_p
    dir_map = [(1, 0), (1, -1), (0, -1), (-1, -1),
               (-1, 0), (-1, 1), (0, 1), (1, 1)]     # 方向在(x, y)的增量
    while now_p != start_p:
        route[now_p[1], now_p[0]] = True
        diretion = close_list[now_p][1]
        now_p = (now_p[0] + dir_map[diretion][0], now_p[1] + dir_map[diretion][1])
    route[start_p[1], start_p[0]] = True
    return close_list[end_p][0], route


def is_valid(point, raster, closed_list):
    '''判断传入point=(x,y)坐标下，在raster内是否为合法的点：在raster范围内，不是closed状态，且不是nan'''
    return 0 <= point[0] < raster.shape[1] and 0 <= point[1] < raster.shape[0] \
        and point not in closed_list and not np.isnan(raster[point[1], point[0]])


def manhattan_distance(p1, p2, mean_cost):
    '''计算两个点的曼哈顿距离，用于rock邻近预估两个点的成本'''
    return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) * mean_cost


def queen_distance(p1, p2, mean_cost):
    '''计算两个点的queen步距离，用于queen邻近预估两个点的成本'''
    delta_x = abs(p1[0] - p2[0])
    delta_y = abs(p1[1] - p2[1])
    return (min(delta_x, delta_y) * SQRT2 + abs(delta_x - delta_y)) * mean_cost
