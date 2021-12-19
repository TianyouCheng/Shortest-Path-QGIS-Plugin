from os import name
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib as mpl
from osgeo import gdal
import os.path as op
import sys
import dijkstra

filename='data/dem.tif'
def read_geotiff(filename):
    tiff=gdal.Open(filename)
    row=tiff.RasterXSize
    col=tiff.RasterYSize
    band=tiff.RasterCount
    geoTransform=tiff.GetGeoTransform()
    data=tiff.ReadAsArray(0,0,row,col)
    proj=tiff.GetProjection()
    return data,row,col,band,geoTransform,proj

def write_geotiff(filename,data,**kwargs):
    driver=gdal.GetDriverByName('GTiff')
    dataset=driver.Create(filename,kwargs['row'],kwargs['col'],1,gdal.GDT_UInt32)
    dataset.SetGeoTransform(kwargs['geoTransform'])
    dataset.SetProjection(kwargs['proj'])
    dataset.GetRasterBand(1).WriteArray(data)
    del dataset
    

if __name__=='__main__':
    filename=op.join(sys.path[0],'data/dem_new.tif')
    cost_raster=read_geotiff(filename)
    # rslt=dijkstra.dijkstra(((2550,540),),(((2586,773),),),cost_raster,True)
    # color1 = colorConverter.to_rgba('white',alpha=0.0)
    # color2 = colorConverter.to_rgba('red',alpha=1)
    # cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    # path_np=np.zeros(cost_raster.shape,dtype=np.uint8)
    
    # for path in rslt[0][0]:
    #     path_np[path[1],path[0]]=10000
    # print(rslt)
    # plt.imshow(cost_raster)
    # plt.imshow(path_np,cmap=cmap2)
    # plt.show()
    data=cost_raster[0].astype(np.int32)
    data=(data+data.min())**2
    write_geotiff(
        op.join(sys.path[0],'data/dem_new.tif'),data,
        row=cost_raster[1],
        col=cost_raster[2],
        geoTransform=cost_raster[4],
        proj=cost_raster[5]
    )
    print(1)