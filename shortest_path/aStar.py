# 使用A_star算法计算最短路径

__all__ = ('a_star',)

import numpy as np
import math
import heapq

SQRT2 = math.sqrt(2)
SQRT_HALF = math.sqrt(0.5)


class PointData:
    '''一个记录点坐标、最短距离、估计距离的类'''
    def __init__(self, point, min_dist, est_dist, direction):
        self.point = point          # 点的坐标，行列号
        self.min_dist = min_dist    # 该点到起点的最短距离
        self.est_dist = est_dist    # 该点到终点的估计距离
        # 回溯方向是0~7的整数: 0表示向东，1表示东南，每+1表示顺时针增大45°
        self.direction = direction  # 该点的回溯方向
        self.total_dist = self.min_dist + self.est_dist

    def update_data(self, min_dist, direction):
        self.min_dist = min_dist
        self.total_dist = self.min_dist + self.est_dist
        self.direction = direction

    def __lt__(self, other):
        return self.total_dist < other.total_dist

    def __hash__(self):
        return hash(self.point)


def a_star(raster: np.ndarray, start_p, end_p, walk_type, progressBar=None):
    '''
    使用a*算法（a_star）算法计算最短路径。
    到终点的估计距离 = 该点到终点经过的格子数（queen步为对角距离、rook步为曼哈顿距离） * raster平均成本

    :param raster: 成本栅格，numpy的矩阵格式
    :param start_p: 起点的点坐标(x, y)，已经转化为行列编号：x是第x列（从0开始），y是"从上往下"第y行（从0开始）
    :param end_p: 终点的点坐标(x, y)，同上
    :param walk_type: str, 表示是`queen邻近`还是`rook邻近`
    :returns: (最低成本距离, 路径列表)，路径以list列表形式返回，每个元素是经过的栅格点行列号(x, y)
    '''
    mean_cost = np.nanmean(raster[min(start_p[0], end_p[0]):(max(start_p[0], end_p[0]) + 1),
                                  min(start_p[1], end_p[1]):(max(start_p[1], end_p[1]) + 1)])
    # mean_cost = np.nanquantile(raster[min(start_p[0], end_p[0]):(max(start_p[0], end_p[0]) + 1),
    #                               min(start_p[1], end_p[1]):(max(start_p[1], end_p[1]) + 1)], q=0.1)
    is_queen = walk_type[:5].lower() == 'queen'
    est_dist_func = queen_distance if is_queen else manhattan_distance
    est_total = est_dist_func(start_p, end_p, mean_cost)  # 估计总距离多长，用于进度条更新
    # 开放列表：list，实际运算时需要用heap方法加速计算最小值
    start = PointData(start_p, 0, est_total, None)
    open_list = [start]
    open_dict = {start_p: start}
    close_list = {}   # 封闭列表，查表找到点的数据

    # 循环，计算最小成本，和每个格子的回溯路径
    while True:
        now_pdata = heapq.heappop(open_list)
        now_p, value = now_pdata.point, now_pdata.min_dist
        if now_p not in open_dict:
            continue
        open_dict.pop(now_p)
        close_list[now_p] = now_pdata
        if progressBar is not None:
            bar_value = int((est_total - now_pdata.est_dist) * 100 / est_total)
            if bar_value > progressBar.value():
                progressBar.setValue(bar_value if bar_value > 0 else 0)
        if now_p == end_p:
            break
        # nearby的内容为(点坐标: 回溯方向)
        nearby = [((now_p[0] - 1, now_p[1]), 0), ((now_p[0], now_p[1] + 1), 2),
                  ((now_p[0] + 1, now_p[1]), 4), ((now_p[0], now_p[1] - 1), 6)]
        if is_queen:
            nearby.extend([((now_p[0] - 1, now_p[1] + 1), 1), ((now_p[0] + 1, now_p[1] + 1), 3),
                           ((now_p[0] + 1, now_p[1] - 1), 5), ((now_p[0] - 1, now_p[1] - 1), 7)])
        for k, v in nearby:
            if not is_valid(k, raster, close_list):
                continue
            min_dist = value + (raster[now_p] + raster[k]) * \
                       (0.5 if v % 2 == 0 else SQRT_HALF)
            if k not in open_dict or open_dict[k].min_dist > min_dist:
                est_dist = est_dist_func(k, end_p, mean_cost) if k not in open_dict \
                    else open_dict[k].est_dist
                pdata = PointData(k, min_dist, est_dist, v)
                heapq.heappush(open_list, pdata)
                open_dict[k] = pdata

    # 从close_list计算回溯路径
    route_list = []
    now_p = end_p
    dir_map = [(1, 0), (1, -1), (0, -1), (-1, -1),
               (-1, 0), (-1, 1), (0, 1), (1, 1)]     # 方向在(x, y)的增量
    while now_p != start_p:
        route_list.append(now_p)
        direction = close_list[now_p].direction
        now_p = (now_p[0] + dir_map[direction][0], now_p[1] + dir_map[direction][1])
    route_list.append(start_p)
    route_list.reverse()
    return close_list[end_p].min_dist, route_list


def is_valid(point, raster, closed_list):
    '''判断传入point=(x,y)坐标下，在raster内是否为合法的点：在raster范围内，不是closed状态，且不是nan'''
    return point not in closed_list and 0 <= point[0] < raster.shape[0] \
           and 0 <= point[1] < raster.shape[1] and not np.isnan(raster[point])


def manhattan_distance(p1, p2, mean_cost):
    '''计算两个点的曼哈顿距离，用于rock邻近预估两个点的成本'''
    return (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) * mean_cost


def queen_distance(p1, p2, mean_cost):
    '''计算两个点的queen步距离，用于queen邻近预估两个点的成本'''
    delta_x = abs(p1[0] - p2[0])
    delta_y = abs(p1[1] - p2[1])
    return (min(delta_x, delta_y) * SQRT2 + abs(delta_x - delta_y)) * mean_cost


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from osgeo import gdal

    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    import time
    from coord_to_num import coord_to_num
    ds = gdal.Open(r'data/dem.tif')
    raster = np.abs(ds.ReadAsArray())
    # cost = (raster + np.min(raster)).astype(np.int64) ** 3
    t = time.time()
    src_p = coord_to_num(ds, 116.9, 40.9)
    end_p = coord_to_num(ds, 116.5, 40.7)
    print('src:', src_p, 'end:', end_p)
    distance, route_list = a_star(raster, src_p, end_p, 'queen邻近')
    print('time:', time.time() - t)
    print('cost:', distance)

    plt.figure(figsize=(8, 6))
    plt.imshow(raster)
    plt.title('路径')
    plt.colorbar()
    xy = list(zip(*route_list))
    plt.plot(xy[0], xy[1], color='red')
    # plt.xlim(1000, 1500)
    # plt.ylim(1200, 700)
    plt.show()
