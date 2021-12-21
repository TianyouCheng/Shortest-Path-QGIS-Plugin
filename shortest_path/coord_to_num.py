import numpy as np
from osgeo import gdal
import math

def coord_to_num(dataset, x, y):
    '''将投影坐标转成行列号(x, y)'''
    geotransform = dataset.GetGeoTransform()
    b = np.array([x - geotransform[0], y - geotransform[3]])
    A = np.array([[geotransform[1], geotransform[2]], [geotransform[4], geotransform[5]]])
    x_pixel, y_line = [int(i) for i in np.linalg.solve(A, b)]
    return  y_line, x_pixel


def num_to_coord(dataset, px, line):
    '''将行列号转成投影坐标(lng, lat)或(x, y)'''
    px, line = line, px
    geotransform = dataset.GetGeoTransform()
    x = geotransform[0] + (px + 0.5) * geotransform[1] + (line + 0.5) * geotransform[2]
    y = geotransform[3] + (px + 0.5) * geotransform[4] + (line + 0.5) * geotransform[5]
    return x, y
