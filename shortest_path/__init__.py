# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ShortestPath
                                 A QGIS plugin
 This plugin can create a shortest path
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2021-11-21
        copyright            : (C) 2021 by Cheng. Hu. Hou. Jiang.
        email                : 2101210065@stu.pku.edu.cn
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load ShortestPath class from file ShortestPath.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .shortest_path import ShortestPath
    return ShortestPath(iface)
