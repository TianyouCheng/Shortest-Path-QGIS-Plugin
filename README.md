# Shortest-Path-QGIS-Plugin
For GIS Presentation

## 新增说明：“机器码加速”模式介绍
### 说明
    使用numba库对代码进行装饰，编译为机器码执行。
### 安装方法
    要在QGIS的python里安装库，在所有程序中搜OSGeo4W Shell，以管理员身份打开，可以安装第三方库。pip install -U numba
### numba编译特点
    numba支持有限的python结构，仅支持numpy库。因此程序中无法传入进度条对象。
    目前去除了堆结构，如果自己实现，可以达到更加快速的效果。
### numba官方文档
    https://numba.readthedocs.io/en/stable/developer/index.html

## 参考文档
----
### 输入输出介绍
    https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-analyst-toolbox/understanding-cost-distance-analysis.htm
### 官方算法介绍
    https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-analyst-toolbox/how-the-cost-distance-tools-work.htm

## 成本距离输入
----
### 路径计算方式
    Queen——八邻接
    Rook——四邻接
### 成本输入
    成本栅格可以是单个栅格，且通常都是多个栅格组合的结果。为成本栅格指定的单位可以是任何所需成本类型：金钱成本、时间、能量消耗、或相对于分配给其他像元的成本而得出其含义的无单位系统。输入成本栅格上的值可以是整型或浮点型，但不可以是负值或 0（成本不能为负或为零）。成本栅格不可以包含值 0，因为该算法是一个乘法过程。
    提示: 如果成本栅格中的确包含表示成本最低区域的值 0，那么请在运行成本距离前将这些值更改为某个较小的正值（如 0.01）。可使用条件函数工具执行此操作。如果值 0 表示的是应从分析中排除的区域，那么应在运行成本距离前通过运行设为空函数将这些值更改为 NoData。
### 最大距离
    定义阈值使得累积的成本值不能超过该阈值。如果累计成本值超过了该阈值，对应输出部分的单元将变为NoData。它定义了成本进行累积计算的范围。
