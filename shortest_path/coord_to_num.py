import numpy as np
from osgeo import gdal
import math

def coord_to_num(dataset, x, y):
    x_size = dataset.RasterXSize()
    y_size = dataset.RasterYSize()
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()
    b = np.array([x - geotransform[0], y - geotransform[3]])
    A = [[geotransform[1], geotransform[2]], [geotransform[4], geotransform[5]]]
    x_pixel, y_line = [int(i) for i in np.linalg().solve(A, b)]
    return x_pixel, y_size - 1 - y_line
