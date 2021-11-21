# Shortest-Path-QGIS-Plugin
For GIS Presentation

## 参考文档
----
### 输入输出介绍
    https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-analyst-toolbox/understanding-cost-distance-analysis.htm
### 官方算法介绍
    https://desktop.arcgis.com/zh-cn/arcmap/10.3/tools/spatial-analyst-toolbox/how-the-cost-distance-tools-work.htm

## 成本距离输入
----
### 源输入
    如果源数据集是一个栅格数据，它可能包含单个或多个区域。这些区域可以相连，也可以不相连。所有具有值（包括 0）的像元都将作为源像元进行处理。源栅格中的所有非源像元都必须赋予值 NoData。而分配给源位置（栅格或要素）的原始值则得以保留。
    如果源数据集是一个要素数据集，则会在内部将其转换为栅格，而该栅格的分辨率将由环境决定；如果并未明确设置分辨率，则将采用与输入成本栅格相同的分辨率。如果源数据是一个栅格数据，则会使用源栅格的像元大小。从此处开始，本文档将假设已将要素源数据转换为栅格数据。
    不存在任何对于输入栅格或要素源数据中源的数量的固有限制。
### 源字段
    对于矢量数据，必须包含至少一个有效的字段。它用来获取源的值。
### 源计算类型
    定义输入目标数据上的值和区域在成本路径计算中的解释方式的关键字。
    EACH_CELL — For each cell with valid values on the input destination data, a least-cost path is determined and saved on the output raster. With this option, each cell of the input destination data is treated separately, and a least-cost path is determined for each from cell.
    EACH_ZONE — For each zone on the input destination data, a least-cost path is determined and saved on the output raster. With this option, the least-cost path for each zone begins at the cell with the lowest cost distance weighting in the zone.
    BEST_SINGLE — For all cells on the input destination data, the least-cost path is derived from the cell with the minimum of the least-cost paths to source cells.
### 路径计算方式
    Queen——八邻接
    Rook——四邻接
    Bishop——四邻接，只能走对角
### 成本输入
    成本栅格可以是单个栅格，且通常都是多个栅格组合的结果。为成本栅格指定的单位可以是任何所需成本类型：金钱成本、时间、能量消耗、或相对于分配给其他像元的成本而得出其含义的无单位系统。输入成本栅格上的值可以是整型或浮点型，但不可以是负值或 0（成本不能为负或为零）。成本栅格不可以包含值 0，因为该算法是一个乘法过程。
    提示: 如果成本栅格中的确包含表示成本最低区域的值 0，那么请在运行成本距离前将这些值更改为某个较小的正值（如 0.01）。可使用条件函数工具执行此操作。如果值 0 表示的是应从分析中排除的区域，那么应在运行成本距离前通过运行设为空函数将这些值更改为 NoData。
### 最大距离
    定义阈值使得累积的成本值不能超过该阈值。如果累计成本值超过了该阈值，对应输出部分的单元将变为NoData。它定义了成本进行累积计算的范围。
