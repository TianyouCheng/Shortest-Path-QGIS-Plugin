# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ShortestPathDialog
                                 A QGIS plugin
 This plugin can create a shortest path
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2021-11-21
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Cheng. Hu. Hou. Jiang.
        email                : 2101210065@stu.pku.edu.cn
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets
from qgis.core import QgsVectorLayer,QgsProject,QgsRasterLayer
import qgis.utils
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from osgeo import ogr
from osgeo import gdal
import numpy as np
from .coord_to_num import coord_to_num, num_to_coord
from .aStar import a_star
# from .aStarCTY import a_starCTY
from .astarCTY import a_starCTY
import time


# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'shortest_path_dialog_base.ui'))


class ShortestPathDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(ShortestPathDialog, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect

        self.setupUi(self)

        # 信号与槽函数
        self.bt_loadcost.clicked.connect(lambda: self.initlabel(self.line_cost))
        self.bt_export.clicked.connect(lambda: self.initlabel(self.line_export))
        self.bt_cancel.clicked.connect(self.close)
        # self.bt_ok.clicked.connect(self.ProcessMindis)
        self.bt_ok.clicked.connect(self.MainProcess)

    def initlabel(self,initobj):
        if initobj==self.line_export:
            ofd, filt = QFileDialog.getSaveFileName(self, '选择shapefile文件', './', 'Shapefile文件(*.shp);;ALL(*.*)')
            initobj.setText(ofd)
        else:
            ofd, filt = QFileDialog.getOpenFileName(self, '选择TIF文件', './', 'TIF文件(*.tif);;ALL(*.*)')
            initobj.setText(ofd)

    def MainProcess(self):
        # 栅格文件读取到数组
        path = self.line_cost.text()    # TIF文件路径
        ds = gdal.Open(path)
        if ds is None:
            self.textResult.setPlainText('Error:\n\ncost raster NOT found!')
            return
        band1 = np.abs(ds.ReadAsArray())
        nearby = self.cb_path.currentText()     # 邻接方式的文本

        # 起止点坐标 TODO:判断为int型，判断在图幅内，弹出报错
        # print(int(self.srcX.text()))
        # print(int(self.srcY.text()))
        # print(int(self.endX.text()))
        # print(int(self.endY.text()))
        try:
            src_p = coord_to_num(ds, float(self.srcX.text()), float(self.srcY.text()))
            end_p = coord_to_num(ds, float(self.endX.text()), float(self.endY.text()))
        except ValueError:
            self.textResult.setPlainText('Error:\n\npoints coordinate must be number!')
            return
        if not (0 <= src_p[0] < band1.shape[0] and 0 <= src_p[1] < band1.shape[1]
                and 0 <= end_p[0] < band1.shape[0] and 0 <= end_p[1] < band1.shape[1]):
            self.textResult.setPlainText('Error:\n\nstart point or end point out of raster!')
            return

        # 进度条使用。进度条的取值范围为0-100
        self.progressBar.setValue(0)

        thread = MainWorkThread(band1, src_p, end_p, nearby,
                                parent=self, progressBar=self.progressBar,fast_mode=self.checkBoxs_FastMode.isChecked(),compile=self.checkBox_MachineLang.isChecked())

        def mainwork_finish(min_dist, route_list):
            '''计算完成后的操作'''
            self.tabWidget.setCurrentIndex(1)  # 激活“结果”选项卡
            self.textResult.setPlainText(
                'Minimum cost distance:{}\n\nTime use:{} seconds'.format(min_dist, time.time() - start_time))  # 写入结果
            self.progressBar.setValue(100)  # 进度条进度

            # 从数组写回shp文件，表示求解得到的路径
            output_path = self.line_export.text()  # 输出文件路径
            if output_path:

                oDriver = ogr.GetDriverByName('ESRI Shapefile')
                oDs = oDriver.CreateDataSource(output_path)
                if os.path.exists(output_path):
                    oDriver.DeleteDataSource(output_path)
                outlayer = oDs.CreateLayer(os.path.splitext(output_path)[0], srs=ds.GetSpatialRef(),
                           geom_type=ogr.wkbLineString)
                outlayer.CreateField(ogr.FieldDefn('cost', ogr.OFTReal))
                route_ = list(map(lambda p: num_to_coord(ds, p[0], p[1]), route_list))
                geom = ogr.CreateGeometryFromWkt(f"LINESTRING({','.join([f'{p[0]} {p[1]}' for p in route_])})")
                ft = ogr.Feature(outlayer.GetLayerDefn())
                ft.SetGeometry(geom)
                ft.SetField('cost', min_dist)
                outlayer.CreateFeature(ft)
                outlayer.SyncToDisk()

                # shp文件加载到QGIS中
                layer_name = os.path.splitext(os.path.basename(output_path))[0]
                layer = QgsVectorLayer(output_path, layer_name, 'ogr')
                QgsProject.instance().addMapLayer(layer)

        thread.complete.connect(mainwork_finish)
        start_time = time.time()
        thread.start()


class MainWorkThread(QThread):
    '''执行主函数的线程类'''
    complete = pyqtSignal([float, list], [int, list])

    def __init__(self, band, start_p, end_p, walk_type, parent=None, progressBar=None, fast_mode=False ,compile=False):
        super(MainWorkThread, self).__init__(parent)
        self.raster = band
        self.start_p = start_p
        self.end_p = end_p
        self.work_type = walk_type
        self.pBar = progressBar
        self.fast = fast_mode
        self.compile=compile

    def run(self) -> None:
        if self.compile:
            min_dist, route_list = a_starCTY(self.raster, self.start_p, self.end_p, self.work_type,
                                      self.fast)
            st=time.time()
            min_dist, route_list = a_starCTY(self.raster, self.start_p, self.end_p, self.work_type,
                                             self.fast)
            Scd=time.time() - st
            print('SecondRound Time Use:{}'.format(Scd))
            self.pBar.enabled=False
        else:
            min_dist, route_list =a_star(self.raster, self.start_p, self.end_p, self.work_type,
                                      self.pBar, self.fast)
        self.complete.emit(min_dist, route_list)
