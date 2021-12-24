# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'shortest_path_dialog_base.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ShortestPathDialogBase(object):
    def setupUi(self, ShortestPathDialogBase):
        ShortestPathDialogBase.setObjectName("ShortestPathDialogBase")
        ShortestPathDialogBase.resize(1018, 612)
        self.gridLayout_3 = QtWidgets.QGridLayout(ShortestPathDialogBase)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_5 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_5.setTextFormat(QtCore.Qt.PlainText)
        self.label_5.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 6, 0, 1, 1)
        self.line_export = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.line_export.setObjectName("line_export")
        self.gridLayout_2.addWidget(self.line_export, 7, 0, 1, 2)
        self.cb_path = QtWidgets.QComboBox(ShortestPathDialogBase)
        self.cb_path.setObjectName("cb_path")
        self.cb_path.addItem("")
        self.cb_path.addItem("")
        self.gridLayout_2.addWidget(self.cb_path, 5, 0, 1, 2)
        self.checkBox_MachineLang = QtWidgets.QCheckBox(ShortestPathDialogBase)
        self.checkBox_MachineLang.setObjectName("checkBox_MachineLang")
        self.gridLayout_2.addWidget(self.checkBox_MachineLang, 9, 1, 1, 1)
        self.label = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label.setLineWidth(1)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_9 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_9.setTextFormat(QtCore.Qt.PlainText)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 0, 1, 1)
        self.srcX = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.srcX.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.srcX.setObjectName("srcX")
        self.gridLayout.addWidget(self.srcX, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_10.setTextFormat(QtCore.Qt.PlainText)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 2, 1, 1)
        self.srcY = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.srcY.setObjectName("srcY")
        self.gridLayout.addWidget(self.srcY, 0, 3, 1, 1)
        self.label_11 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_11.setTextFormat(QtCore.Qt.PlainText)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 0, 4, 1, 1)
        self.endX = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.endX.setObjectName("endX")
        self.gridLayout.addWidget(self.endX, 0, 5, 1, 1)
        self.label_12 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_12.setTextFormat(QtCore.Qt.PlainText)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 0, 6, 1, 1)
        self.endY = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.endY.setObjectName("endY")
        self.gridLayout.addWidget(self.endY, 0, 7, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 3)
        self.line_cost = QtWidgets.QLineEdit(ShortestPathDialogBase)
        self.line_cost.setObjectName("line_cost")
        self.gridLayout_2.addWidget(self.line_cost, 3, 0, 1, 2)
        self.checkBoxs_FastMode = QtWidgets.QCheckBox(ShortestPathDialogBase)
        self.checkBoxs_FastMode.setObjectName("checkBoxs_FastMode")
        self.gridLayout_2.addWidget(self.checkBoxs_FastMode, 9, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_7.setTextFormat(QtCore.Qt.PlainText)
        self.label_7.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 4, 0, 1, 2)
        self.bt_loadcost = QtWidgets.QPushButton(ShortestPathDialogBase)
        self.bt_loadcost.setMaximumSize(QtCore.QSize(80, 16777215))
        self.bt_loadcost.setObjectName("bt_loadcost")
        self.gridLayout_2.addWidget(self.bt_loadcost, 3, 2, 1, 1)
        self.bt_export = QtWidgets.QPushButton(ShortestPathDialogBase)
        self.bt_export.setMaximumSize(QtCore.QSize(80, 16777215))
        self.bt_export.setObjectName("bt_export")
        self.gridLayout_2.addWidget(self.bt_export, 7, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(ShortestPathDialogBase)
        self.label_3.setTextFormat(QtCore.Qt.PlainText)
        self.label_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 0, 1, 2)
        self.tabWidget = QtWidgets.QTabWidget(ShortestPathDialogBase)
        self.tabWidget.setMinimumSize(QtCore.QSize(300, 0))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textDescript = QtWidgets.QTextEdit(self.tab_3)
        self.textDescript.setObjectName("textDescript")
        self.verticalLayout.addWidget(self.textDescript)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textResult = QtWidgets.QTextEdit(self.tab_4)
        self.textResult.setObjectName("textResult")
        self.verticalLayout_2.addWidget(self.textResult)
        self.tabWidget.addTab(self.tab_4, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 3, 10, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 8, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 2)
        self.progressBar = QtWidgets.QProgressBar(ShortestPathDialogBase)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_3.addWidget(self.progressBar, 1, 0, 1, 2)
        self.bt_ok = QtWidgets.QPushButton(ShortestPathDialogBase)
        self.bt_ok.setMinimumSize(QtCore.QSize(0, 30))
        self.bt_ok.setObjectName("bt_ok")
        self.gridLayout_3.addWidget(self.bt_ok, 2, 0, 1, 1)
        self.bt_cancel = QtWidgets.QPushButton(ShortestPathDialogBase)
        self.bt_cancel.setMinimumSize(QtCore.QSize(0, 30))
        self.bt_cancel.setObjectName("bt_cancel")
        self.gridLayout_3.addWidget(self.bt_cancel, 2, 1, 1, 1)

        self.retranslateUi(ShortestPathDialogBase)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ShortestPathDialogBase)

    def retranslateUi(self, ShortestPathDialogBase):
        _translate = QtCore.QCoreApplication.translate
        ShortestPathDialogBase.setWindowTitle(_translate("ShortestPathDialogBase", "Shortest Path"))
        self.label_5.setText(_translate("ShortestPathDialogBase", "输出栅格"))
        self.cb_path.setItemText(0, _translate("ShortestPathDialogBase", "Queen邻接"))
        self.cb_path.setItemText(1, _translate("ShortestPathDialogBase", "Rook邻接"))
        self.checkBox_MachineLang.setText(_translate("ShortestPathDialogBase", "机器码加速"))
        self.label.setText(_translate("ShortestPathDialogBase", "起止点横纵坐标"))
        self.label_9.setText(_translate("ShortestPathDialogBase", "起点x"))
        self.label_10.setText(_translate("ShortestPathDialogBase", "起点y"))
        self.label_11.setText(_translate("ShortestPathDialogBase", "终点x"))
        self.label_12.setText(_translate("ShortestPathDialogBase", "终点y"))
        self.checkBoxs_FastMode.setText(_translate("ShortestPathDialogBase", "快速模式"))
        self.label_7.setText(_translate("ShortestPathDialogBase", "路径计算方式（可选）"))
        self.bt_loadcost.setText(_translate("ShortestPathDialogBase", "..."))
        self.bt_export.setText(_translate("ShortestPathDialogBase", "..."))
        self.label_3.setText(_translate("ShortestPathDialogBase", "成本输入（栅格数据）"))
        self.textDescript.setHtml(_translate("ShortestPathDialogBase", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">计算成本最低路径</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">开发人员：第二小组（程天佑 胡俊杰 姜金廷 侯远樵）</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">主要功能：基于栅格成本图层，根据用户输入的起始点坐标和终止点坐标计算成本，并显示计算耗时</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">参数说明：</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">起点x，起点y</span>——成本路径起点的经度和纬度，需要用户手动输入（请注意顺序）</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">终点x，终点y</span>——成本路径终点的经度和纬度，需要用户手动输入（请注意顺序）</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">成本输入</span>——用户点击...按钮，系统自动弹出对话框提示用户选择成本栅格所在路径，并将用户选择的路径显示在文本框内</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">连接方式</span>——用户可以选择“queen”和“rook”两种模式，前者模式下，最终生成的路径可以出现对角斜线；后者模式下，最终生成的路径只能按照水平方向或者竖直方向移动</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">输出栅格</span>——和成本输入类似，用户在弹出的对话框中手动选择输出栅格的路径和位置</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">快速模式——用于提高路径的计算速度，但可能会导致成本上升，无法得到最优解，为可选项</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">机器码加速——使用机器码技术实现的更高效的路径计算方法，为可选项</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">结果输出：</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1）弹出图像显示算法得到的成本最低路径总成本</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2）在“结果”栏显示得到的最优路径总成本以及运算时间</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">其他：在计算过程中，可以实时显示进度</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("ShortestPathDialogBase", "描述"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("ShortestPathDialogBase", "结果"))
        self.bt_ok.setText(_translate("ShortestPathDialogBase", "确定"))
        self.bt_cancel.setText(_translate("ShortestPathDialogBase", "取消"))
