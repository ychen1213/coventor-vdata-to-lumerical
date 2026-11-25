# AH53 长侧壁拉直与局部减面 · AH53 sidewall straightening

这个示例把凸起硅条两侧的长边拉直，用来比较侧壁起伏对光学模型的影响。在每个选定高度，程序分别对每条边拟合直线 `x = a(z) + b(z)y`。截距保留这一高度的平均边缘位置，斜率保留沿条纹长度方向的整体倾斜。各高度分别拟合，因此侧壁可以继续随深度倾斜。程序在改动区域的边界接回原网格，保留范围外的表面。

## 输入和运行方法

输入是本仓库独立转换器生成的 **单一硅实体 VMESH**。示例采用三条平行硅条、六条长边的 AH53 预设。先核对 `straighten.py` 中的边缘位置、Y 范围和高度窗口与输入的晶圆坐标是否对应。程序要求在每个采样高度沿 Y 能唯一定位一条边；使用不同布局时需要适配选区和采样位置。

在仓库根目录执行下列命令。把 `inputs/mark.vdata` 换成自己的输入文件，为每一步选择新的输出目录。已安装仓库依赖且已有合适 VMESH 时，从第二段开始，并将输入路径改为自己的网格。

```powershell
python -m pip install -r requirements.txt
python run.py convert inputs/mark.vdata --output outputs/A --surface fraction --storage compact --planar-optimize

python -m examples.ah53_sidewall_straightening.straighten --input outputs/A/meshes/material_0002.vmesh --output outputs/straight

python -m examples.ah53_sidewall_straightening.export_cad --input outputs/straight/B.vmesh --output outputs/B_cad
python -m examples.ah53_sidewall_straightening.export_cad --input outputs/straight/C.vmesh --output outputs/C_cad
```

拉直步骤生成 `B.vmesh`、`C.vmesh`，以及各自的 `B_geometry.json`、`C_geometry.json`。CAD 导出步骤调用 v0.4 的严格共面合并和顶点清理，在每个 CAD 输出目录生成 `geometry.mat` 和 `import_lumerical_coplanar_compact.lsf`。在 Lumerical FDTD 的空白 Layout 中打开并运行对应 LSF，即可建立几何并保存 FSP。默认材料为 `Si (Silicon) - Palik`，可用 `--material` 指定本机材料库名称。若要导入 A，用同一个 `export_cad` 命令传入原始 VMESH。 [独立转换器使用说明](../../README.md) · [共面导出教程](../../docs/coplanar.html)

## 怎样解释比较结果

| 比较 | 回答的问题 |
|---|---|
| **A → B** | A 是输入的粗糙表面，B 是用密网格表示的拉直表面。两者的差异包括长边拉直以及接回原边界所形成的过渡带。 |
| **B → C** | B、C 使用相同的拟合直线和过渡带。C 在侧壁中央用较少的三角形。用这对模型检查局部减面额外引入的几何或光学差异。 |
| **A → C** | 比较原始模型与最终模型，可测量整个流程对文件大小和 CAD 导入耗时的影响。 |

过渡带值得单独看一张 XZ 纵剖图。若边缘要在很小的高度内横移以接回原表面，连接处会形成接近水平的折面。例如横移量是过渡高度的 7 倍时，截面与竖直方向的夹角约为 **82°**。因此解释 A 与 C 的光学差异时，也要考虑连接处的形貌变化。

## English quick guide

The commands above run from the repository root. Supply a single silicon VMESH from the standalone converter. The preset targets six long edges on three raised, Y-oriented bars. Check the edge coordinates and the Y and Z windows in `straighten.py` against your mesh. Each sampled wall must have one identifiable X crossing at every sampled Y. Adapt the selection for other layouts.

At each sampled height the program fits a separate straight line to each edge, preserving its mean position and overall tilt along Y. **A** is the input mesh. **B** represents the straightened walls densely. **C** uses fewer triangles in the wall interiors and shares B’s fitted lines and transition bands. Compare **B with C** to isolate the effect of local mesh reduction. Compare **A with C** to measure the total change in geometry, file size or import time. A narrow transition can create an almost horizontal surface. A lateral displacement seven times the transition height gives an XZ angle of about **82° from vertical**.

For each variant, `export_cad` writes a compact polygon MAT and an import LSF. Run the LSF in a new Lumerical FDTD layout, then save the project as an FSP. Use the same material and import method for all variants.
