from os import name
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from osgeo import gdal
import os.path as op
import sys

filename='data/dem.tif'
def read_geotiff(filename):
    tiff=gdal.Open(filename)
    row=tiff.RasterXSize
    col=tiff.RasterYSize
    band=tiff.RasterCount
    geoTransform=tiff.GetGeoTransform()
    data=np.zeros([row,col,band])
    for i in range(band):
        dt=tiff.GetRasterBand(1+i)
        data[:,:,i]=dt.ReadAsArray(0,0,row,col)
    return data



if __name__=='__main__':
    filename=op.join(sys.path[0],'data/dem.tif')
    cost_raster=read_geotiff(filename)
    plt.imshow(cost_raster)
    plt.show()