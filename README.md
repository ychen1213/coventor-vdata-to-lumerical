# Coventor VDATA → Lumerical FDTD

独立读取 Coventor 导出的 **VoxelXML `.vdata`**，提取按材料划分的表面，生成 STL、光学材料映射和 Lumerical 导入脚本；可用已安装的 FDTD 保存并重新打开 `.fsp` 核验。

几何来自输入 VDATA，不需要原始工艺工程、版图或工艺参数，不重新运行工艺仿真。

这个目录只包含程序源码、启动器、依赖清单与本说明。没有模型、仿真结果、缓存、第三方安装包、Coventor/Ansys SDK 或许可证。

## 提供的版本

| 模式 | 入口 | 行为 |
| --- | --- | --- |
| 不减面 | `convert_vdata.py` | 保留原生表面导出所得的全部三角面 |
| 减面45% | `convert_reduced.py --reduction 0.45` | 转换后执行 QEM，目标保留约55%的面 |
| 减面65% | `convert_reduced.py --reduction 0.65` | 转换后执行 QEM，目标保留约35%的面 |
| 自定义比例 | `convert_reduced.py --reduction 0.30` | 比例为删除的面数占比，取值 `[0,1)`；0走不减面路线 |
| 对已有转换结果减面 | `reduce_conversion.py` | 复用 `conversion_manifest.json`，不重复原生 VDATA 网格导出 |

共用底层代码，不复制数份转换器。**不减面不是 VDATA 边界的无损重建**：原生导出的表面本身仍可能是近似。

## 运行条件

- Windows；不减面入口使用 Python 3.6+ 标准库，图形界面需要 `tkinter`。
- 已安装并获得相应许可的 Coventor SEMulator3D Python SDK。当前已验证 API 9.3 的 VoxelXML 路线；其他编码或版本不保证兼容。
- 请求 `--verify-fdtd` 时，需要已安装的 Ansys Lumerical FDTD 及 CAD 许可。
- QEM入口需要 **Python 3.10+** 及 `requirements-qem.txt` 中的包。使用运行该入口的同一个解释器安装：

```powershell
python -m pip install -r .\requirements-qem.txt
```

SDK由程序在本机发现，不通过pip安装、不随仓库分发。主程序、Coventor SDK和Lumerical验证可使用各自的Python解释器。

## 图形界面

双击 `launch_converter.cmd`，或将 VDATA 文件拖到该启动器上。选择“不减面 / 减面45% / 减面65% / 自定义比例”，再指定新的输出目录。

启动器优先使用环境变量 `VDATA_CONVERTER_PYTHON`，否则寻找可用的系统Python或Ansys自带Python。若选择QEM模式，所选解释器必须安装QEM依赖；缺少依赖会在耗时的原生转换前报错。

```powershell
# 替换为实际安装了QEM依赖和tkinter的Python路径
$env:VDATA_CONVERTER_PYTHON = "C:\Path\To\Python\python.exe"
.\launch_converter.cmd
```

“验证并生成 FDTD 工程”只启动私有隐藏CAD会话，不接管已经打开的工程，也不运行电磁时域求解。

## 命令行示例

在程序目录执行；`python`应指向符合上述条件的解释器。输入、输出均可包含中文或空格。

```powershell
# 只读元数据，不启动有许可证的软件
python .\convert_vdata.py "C:\models\sample.vdata" --inspect

# 不减面，生成并核验FDTD工程
python .\convert_vdata.py "C:\models\sample.vdata" `
  --output "C:\models\sample_full" --verify-fdtd

# 从VDATA直接生成45%减面版本
python .\convert_reduced.py "C:\models\sample.vdata" `
  --output "C:\models\sample_qem45" --reduction 0.45 --verify-fdtd

# 65%减面版本
python .\convert_reduced.py "C:\models\sample.vdata" `
  --output "C:\models\sample_qem65" --reduction 0.65 --verify-fdtd

# 复用已有不减面结果：省去原生VDATA导出阶段
python .\reduce_conversion.py "C:\models\sample_full\conversion_manifest.json" `
  --output "C:\models\sample_reduced" --reduction 0.45 --verify-fdtd
```

省略 `--verify-fdtd` 时只生成几何、映射模板和 `.lsf`，不生成新的 `.fsp`。

软件未能自动发现时，不减面和VDATA直接减面入口可传入：

```powershell
python .\convert_reduced.py "C:\models\sample.vdata" `
  --output "C:\models\sample_qem" --reduction 0.45 --verify-fdtd `
  --semulator-root "C:\Path\To\SEMulator3D" --api-version 9.3 `
  --lumerical-root "C:\Path\To\Lumerical"
```

也支持环境变量 `SEMULATOR_ROOT`、`MEMULATOR_ROOT`、`LUMERICAL_ROOT`。已有许可证环境由子进程继承；程序不读取或上传许可证内容。

## 输出与材料映射

不减面版本在输出目录下写入 `meshes/`、`conversion_manifest.json`、`conversion_report.json` 和 `fdtd/`。

VDATA直接减面版本分别保留：

```text
输出目录/
  conversion_report.json
  full_resolution/       原生转换结果，作为减面的基准
  reduced/               减面网格、误差报告和FDTD导入结果
    reduction_report.json
    conversion_manifest.json
    meshes/
    fdtd/
```

FDTD工程位于对应结果的 `fdtd/imported.fsp`，只在验证通过后生成。STL数值单位是µm，导入FDTD时转换为m，并恢复每个材料对象的坐标平移。

材料不会全部自动设为Si。工具保留材料ID、完整名称及位置；明确的Si和SiO₂名称有保守默认映射，其他材料需要编辑 `fdtd/material_map.template.json`。可使用 `id:18` 等键区分相同名称的不同材料。

完成映射后，用同一个入口、相同输入/输出/比例，加上 `--material-map` 与 `--resume` 继续。复用前核对源文件、网格设置及文件哈希；参数改变或网格被修改时拒绝复用。`--resume`不用于更换减面比例。

## 精度与兼容性边界

- 基础转换支持已验证的VoxelXML格式，并非任意名为`.vdata`的文件。源长度单位µm、nm、Å及非零原点已有验证；不兼容格式会报错。
- 缺少`PlaneEquations`时，无法恢复文件没有保存的体素内精细界面。更密的表面网格或FDTD网格不能创造这部分信息。
- 不减面版本支持多种实体材料；**当前QEM版本仅允许一个实体材料**。多材料共界面独立减面可能产生缝隙或重叠，因此默认拒绝，不把单材料实验包装成通用多材料精度保证。
- QEM复用二次误差边折叠，不追加Gaussian平滑，但仍会移动坐标并改变小特征。45%和65%是面数目标，**不是适合任意输入的精度档位**。
- 减面检查闭合性、方向、退化面、连通分量与Euler特征，并做双向“全顶点＋面中心＋边中点”到三角面的距离采样。结果相对不减面STL，是新增误差，**不是连续全表面的Hausdorff上界**；尚未提供稳健自交证明。
- 可用 `--sampled-max-error-nm 1.0` 拒绝采样距离超标的候选；这个选项不能宣称“任何位置都小于1nm”，也不直接约束CD或粗糙度频谱。
- 不自动把掺杂字段转换为色散光学模型。材料映射通过不等于材料拟合、光学网格或求解已收敛。
- 不自动加入某个mark专用光源、边界和监视器。生成的是通用几何工程；具体光学问题需独立配置。

## 源码分工

- `common.py`、`native_worker.py`、`mesh_tools.py`：独立VDATA暂存、原生SDK提面与几何审计。
- `convert_vdata.py`：不减面流程。
- `qem_tools.py`、`reduce_conversion.py`：通用单材料QEM与误差检查、已有结果减面。
- `convert_reduced.py`：VDATA直接转减面版本的两阶段入口。
- `fdtd_bridge.py`：材料映射、导入脚本、私有CAD保存重开核验。
- `gui.py`、`launch_converter.cmd`：模式选择和桌面入口。

## 方法依据

- [Ansys STL导入接口与单位](https://optics.ansys.com/hc/en-us/articles/360034924733-stlimport-Script-command)
- [Garland–Heckbert QEM方法](https://www.mgarland.org/research/quadrics.html)
- [fast-simplification官方接口](https://pyvista.github.io/fast-simplification/)

第三方Python依赖按各自项目的许可分发；本仓库不包含其源码或二进制。Coventor与Ansys产品需要用户自行安装和授权。
