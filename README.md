# Coventor VDATA → Lumerical：独立转换器与图文教程

将已经导出的 Coventor VoxelXML `.vdata` 读取为材料/占比数组，独立重建三角表面，输出 STL 或索引 MAT 及 Lumerical 导入脚本。支持不减面、无损网格压缩和可选 QEM 减面。

**转换过程无需 Coventor SDK、Coventor 安装或商业许可证，只使用普通 Python 与开源库。** 当前支持已经核验的 VoxelXML 编码，不能承诺读取任意版本、任意内部格式的 `.vdata`。真实 VDATA 文件由使用者提供，也可用仓库的人工小样演练。

当前算法版本 **0.2.0**；**2026-10-27** 整理发布软件和最新教程，本次整理不代表新的算法版本。仓库当前入口已切换为独立方案，旧的 Coventor SDK 导出路线不再作为使用步骤。

## 先安装，再完成第一次转换

以下命令在 Windows PowerShell 中执行。已验证环境为普通 **CPython 3.12**；项目声明支持 Python 3.10+，其他版本和系统仍取决于开源依赖的可用性。没有安装 Coventor 或 Lumerical，也可以完成这里的转换。

```powershell
git clone https://github.com/ychen1213/coventor-vdata-to-lumerical.git
cd coventor-vdata-to-lumerical

py -3.12 -m venv .venv
$toolPython = ".\.venv\Scripts\python.exe"
& $toolPython -m pip install -r requirements.txt
& $toolPython run.py --help
```

也可以在 GitHub 选择 **Code → Download ZIP**，解压后进入包含 `run.py` 的目录，从 `py -3.12 -m venv .venv` 开始执行。无需激活虚拟环境，后续命令都明确使用这个解释器。

没有自己的输入文件时，先生成人工小样。这个步骤仅写入一个用于演练的简单 VDATA，**不执行工艺仿真，也不是实际 AH53**：

```powershell
& $toolPython examples/make_sample.py outputs/sample.vdata
$inputVdata = ".\outputs\sample.vdata"
$outputDir = ".\outputs\sample_compact"
```

处理自己的文件时，把 `$inputVdata` 换成输入的完整路径，把 `$outputDir` 换成新的输出目录，再执行：

```powershell
# 读取文件结构、尺寸、坐标和材料表，不解压体素
& $toolPython run.py inspect $inputVdata

# 根据材料占比重建表面，不减面，以紧凑格式保存
& $toolPython run.py convert $inputVdata `
  --output $outputDir --surface fraction --storage compact
```

输出目录必须不存在或为空。`fraction` 决定怎样推断表面；`compact` 决定怎样保存已经生成的网格，两者控制不同步骤。该命令保留原始材料记录，并生成包含全部三角面的 `.vmesh`；**此时还没有生成 STL**。

接下来按用途选择一条路线：

| 用途 | 下一步命令 | 结果 |
|---|---|---|
| 传统 STL 导入 | `& $toolPython run.py export-stl $outputDir` | 展开 STL，更新 `import_lumerical.lsf`；在 Lumerical 中执行该脚本 |
| 共享顶点的 MAT 导入 | `& $toolPython run.py export-planar $outputDir` | 生成索引 MAT 和 `import_lumerical_planar.lsf`；使用这个 Planar Solid 脚本 |
| 自研程序使用材料数组 | 无需继续导出表面 | 读取 `voxels.npz` 与 `voxel_metadata.json` |
| 自研程序使用紧凑三角网格 | 无需展开为 STL | 使用 `vdata_standalone.storage` 读取 `.vmesh` |

MAT 路线保存顶点和面索引，**不是三维二值体素掩模 MAT**。已通过矩阵往返检查，但本版本尚未完成实际 Lumerical CAD 导入验证，应先在小范围模型上验证。

这些命令不启动 Lumerical、不保存 `.fsp`，也不执行电磁求解。在 Lumerical 中导入几何后，还需设置材料、计算区域、光源、边界、网格和监视器，才能运行仿真。

## 按完整转换过程阅读教程

[完整图文教程](docs/tutorial.html)可离线阅读。GitHub 通常显示 HTML 源码；请下载或克隆仓库，用浏览器打开本地 `docs/tutorial.html`，保留同目录的 `assets`。

教程先建立完整过程，再解释每项改进所在的位置：

```text
已经完成工艺仿真的 VDATA
  → 解码材料、占比与坐标
  → 推断界面，生成顶点和三角面
  → 保存为 VMESH / STL / 索引 MAT
  → 在 Lumerical 中建立材料几何对象
  → 配置仿真并映射到 Yee 网格，随后求解
```

其中有两个具体、连续的推导：

- **体素怎样变成三角形**：用 20 nm、25% 硅的例子，从八个体素中心、边上的插值交点讲到四个顶点、两个三角形，再解释 STL 怎样记录它们。附逐步交互图和可运行的小例子。
- **STL 怎样成为可运行的 Lumerical 模型**：从封闭边界与材料指定，讲到仿真设置、Yee 网格及界面处理，区分几何文件、FSP 工程与求解器数组。

理解基础过程后，再看 `plane` / `vof` 改变了什么、无损存储和 QEM 减面各改变什么、误差数字怎样解释。最后沿同一输入和输出目录完成操作；编码、公式与源码结构放在可展开附录。

## 不减面、减面与其他重建路线

下面沿用 `$toolPython`、`$inputVdata`。每次重建使用不同的新输出目录。

```powershell
# 只解码材料/占比及支持的平面字段，不生成表面
& $toolPython run.py decode $inputVdata --output ".\outputs\voxels_only"

# 直接输出传统 STL + 索引 NPZ，不减面
& $toolPython run.py convert $inputVdata `
  --output ".\outputs\standard" --surface fraction --storage standard

# QEM 是可选功能，先安装额外开源依赖
& $toolPython -m pip install -r requirements-qem.txt

# 删除约 45% 的面，保留约 55%，并执行几何检查
& $toolPython run.py convert $inputVdata `
  --output ".\outputs\qem_45" --surface fraction `
  --reduction 0.45 --storage compact

# 若需要传统 STL，再展开这个已经减面的网格
& $toolPython run.py export-stl ".\outputs\qem_45"

# 实验性：根据占比和邻域估计局部平面
& $toolPython run.py convert $inputVdata `
  --output ".\outputs\vof" --surface vof --storage compact

# 实验性：必须换成包含已支持 PlaneEquations 的输入
# 当前仅支持一个实际实体与 Air 255 的界面
& $toolPython run.py convert "C:\my_data\with_planes.vdata" `
  --output ".\outputs\plane" --surface plane --storage compact

# 已有本程序格式的索引 NPZ，可直接压缩，不重新重建
& $toolPython run.py pack-mesh "existing\material_0002.npz" `
  --output ".\outputs\material_0002.vmesh"
```

`--reduction 0.45` 相对于本程序新建的未减面网格计算，不能直接对照旧 Coventor 原生 STL 的减面比例。QEM 会改变几何；任何比例都可能因退化面、非流形、拓扑变化或误差检查而被拒绝。可加 `--max-sampled-error-nm 1.0`，拒绝新增采样距离超过 1 nm 的结果；这不是严格的全表面误差上界，也不是光学位置精度保证。

默认仍是 `--surface fraction --storage standard --reduction 0`。`compact` 不隐含减面；`both` 同时输出 STL、NPZ 和 VMESH，不适合节省整个目录的空间。可用 `--crop-um XMIN XMAX YMIN YMAX ZMIN ZMAX` 先转换小范围，参数是输入坐标系中的微米坐标，应根据自己的元数据选择。

Windows 下也可把 VDATA 拖到 `run.cmd` 上。安装好仓库内 `.venv` 后，它会使用这个普通 Python，在输入旁创建带时间戳的新目录，采用默认 `fraction + standard` 路线。已有其他普通 Python 时，可设置 `VDATA_STANDALONE_PYTHON` 为其完整路径；需要紧凑存储、裁剪或减面时使用明确命令。

## 输出文件与坐标

单个实际实体材料、输入不含平面字段时，第一次 `compact` 转换通常输出 **6 个文件**：

```text
sample_compact/
  conversion_manifest.json    来源哈希、方法、体积、审计、限制和文件列表
  voxels.npz                  raw[X,Y,Z,3]，选定范围的原始材料记录
  voxel_metadata.json         尺寸、SI 坐标、材料表和编码规则
  meshes/
    material_0002.vmesh       无损压缩的顶点、面索引及平移
  material_map.json            源材料 ID 到光学材料库名称的映射
  import_lumerical.lsf         STL 路线脚本；尚无 STL 时会阻止执行
```

`0002` 是示例材料 ID。多材料会增加网格文件；有支持的平面字段时另存 `planes.npz` 和 `plane_metadata.json`。`--skip-voxels` 可只保留几何与相关清单，第一次转换通常应保留原始记录。

在上述 6 文件基础上：

- `export-stl` 新增一个 STL，并更新原脚本和清单，单材料示例成为 **7 文件**。
- `export-planar` 新增一个 MAT 和一个 Planar Solid 脚本，单材料示例成为 **8 文件**。
- 两种导出都执行则为 **9 文件**。它们是不同交接方式，不要求全部执行。

默认 `standard` 输出以 STL 和索引 NPZ 替代单个 VMESH，同样条件下基础输出为 7 文件。应以 `conversion_manifest.json` 的状态和文件列表确认实际结果。

坐标约定：

- 体素按 `X,Y,Z` 排列，末尾三个通道为 `fraction_code, material_1, material_2`，不是 RGB。
- `origin_m` 是第一个体素的低角点，单位米；中心为 `origin + (index+0.5)*voxel_size`。
- 索引网格保存局部米制坐标和 `translation_m`，相加得到世界坐标。
- STL 数值使用微米；LSF 使用 `1e-6` 缩放并恢复全局平移。STL 本身不保存标准单位、材料或平移，交接时应保留整个目录。

解码期间的 `.npy` 内存映射缓存是临时文件，成功或失败后关闭并移除。

## 改进与已测结果

这里的 MB 按十进制计算。以下来自已有 AH53 案例，不代表所有模型具有相同效果。

| 改进对象 | 已测结果 | 适用边界 |
|---|---|---|
| 不减面的网格存储 | 同一张 4,984,128 面网格：STL 249.21 MB，索引 NPZ 28.55 MB，VMESH 约 **1.91 MB** | VMESH 是压缩归档；1.91 MB 不是原生 STL，也不是整个输出目录大小 |
| 索引 MAT 导出 | 保留相同全部三角面，压缩 MAT 约 **21.03 MB** | 已做矩阵往返检查，尚未完成实际 Lumerical CAD 导入验证 |
| 使用已保存的平面 | 小范围模型中，94 条历史原生射线同 YZ 沿 X 的交点误差 RMS **2.55 → 1.11 nm**，样本最大值 **13.46 → 5.13 nm** | 只评价这些样本，不是全表面误差上界或光学位置误差 |
| VOF 引导重建 | 合成斜平面改善明显；真实 AH53 沿 X 射线 RMS 约 **2.19 nm** | 该真实样例最近表面距离 RMS/P95 略有恶化，总体积一致性也未改善，仍是实验选项 |

VMESH 对生成后的索引网格做逐比特往返检查：float64 顶点位模式、面索引、顺序及平移一致。“无损”以这张网格为参照，不表示恢复 Coventor 原生表面，也不表示保存了 VDATA 所有附加字段。

固定面数的二进制 STL 为 `84 + 50 × 面数` 字节。减少重复记录与可逆压缩能够缩小归档；再次展开为同样面数的 STL 仍会回到相应大小。磁盘大小、转换峰值内存、FDTD 求解器内存是不同指标；相同电磁网格下，几何压缩不会自动缩小 Yee 场数组。

## 表面重建的依据与精度边界

| 模式 | 重建依据 | 当前局限 |
|---|---|---|
| `fraction`，默认 | 把材料体积分数作为体素中心标量，以 0.5 提取等值面 | 格平均占比不能唯一决定界面；中心插值是近似 |
| `vof`，实验性 | 从邻域估计法向，调整局部平面匹配体积占比，再构造界面场 | 方向是估计值；最终表面不保证逐格精确体积守恒 |
| `plane`，实验性 | 读取原有 `PlaneEquations` 的法向与偏移，构造中心距离场并提取零等值面 | 不是逐格精确裁切原平面；仅支持已验证约定及单实体与 Air 255 |

三种方法最终都使用开源 Lewiner marching cubes 生成三角网格。补背景样本使表面闭合，也会在裁剪边界构造人工封口，某些格尺度角点可能圆化或倒角。默认不做 Gaussian 平滑、不使用 QEM；“不减面”仍包含从材料记录重建表面的近似。

已测平面约定为 `n·u=d`，`u` 从体素低角点起算并按格宽归一化，第一材料位于负侧。`plane` 保留原始偏移 `d`，源平面切格体积可能与记录占比不一致，清单会记录差异；该约定不能承诺适用于所有厂商版本。

读取器可处理材料表中的多个材料，默认导出有体积且不是 Air/Void 的材料，也可用 `--materials 2,18` 选择。每种材料独立提取表面，三相交界或极薄层可能有缝隙、重叠或消失，当前不保证严格共形的多材料分区。因此 QEM 目前限制为一个实际导出实体材料。

## 在 Python 里查看实际材料占比

以下读取初次转换输出的原始记录，不依赖 STL，不会二值化或平滑占比：

```python
import json
from pathlib import Path
import numpy as np

folder = Path("outputs/sample_compact")
with np.load(folder / "voxels.npz", allow_pickle=False) as data:
    raw = data["raw"]
meta = json.loads((folder / "voxel_metadata.json").read_text(encoding="utf-8"))

code = raw[..., 0]
f1 = np.where(code == 129, 1.0, code.astype(np.float32) / 128.0)
material_id = 2  # 根据自己的材料表选择，不一定都是 2
fraction = ((raw[..., 1] == material_id) * f1
            + (raw[..., 2] == material_id) * (1 - f1))

print("XYZ shape:", fraction.shape)
print("Origin in metres:", meta["origin_m"])
print("One XY cross section:", fraction[:, :, fraction.shape[2] // 2])
```

`code=129` 是两个 ID 相同的纯材料记录；其他有效记录第一材料占比为 `code/128`。材料/占比字节保存在 `voxels.npz`，支持的平面单独保存在 `planes.npz`。掺杂数组尚不解码；应继续保存原始 VDATA。

## Lumerical 材料、脚本与 FSP

明确的晶硅名称有 Palik 默认映射，这是光学材料假设，应结合研究确认。未知材料写成 `null`，脚本会阻止导入，需先指定本机材料库中的名称。

编辑输出中的 `material_map.json`，例如：

```json
{"materials": {"id:2": "Si (Silicon) - Palik", "id:18": "SiO2 (Glass) - Palik"}}
```

再按所用路线更新脚本，无需重新解码 VDATA：

```powershell
# STL 已生成时，更新 STL 路线材料映射和绝对路径
& $toolPython run.py lsf $outputDir `
  --material-map "$outputDir\material_map.json"

# MAT 路线使用对应命令
& $toolPython run.py export-planar $outputDir `
  --material-map "$outputDir\material_map.json"
```

移动目录后也应重新生成对应脚本。在正常运行的 Lumerical 中执行脚本、检查材料和截面，再保存 FSP。几何还需映射到求解器网格，网格收敛与光学测量精度须另行验证。

转换器不执行 Coventor 工艺仿真，不生成新的刻蚀结果；不导入 `voxelModeler` / `voxelWafer`，不调用 ModelBuilder、VoxelDataImport 或 MeshExport，不读取或请求商业许可证。写出 LSF 不启动 Lumerical，程序没有商业 CAD 验证入口或 `--verify-fdtd` 开关。

## 格式兼容性与严格拒绝

当前支持已核验的 `VoxelXML → Base64 → !zlib! 压缩块 → 三字节材料/占比记录`，以及已测的小端 float32 四通道平面字段。`.vdata` 扩展名本身不足以确定兼容性。

支持 m、cm、mm、µm/um、nm、Å 等长度单位的 SI 换算，原点单位独立解析。读取器检查损坏 XML、外部实体/DOCTYPE、未知编码声明、材料 ID 重复/缺失、非法占比、压缩块长度、zlib 结束与尾随数据、总字节数及内存限额；不理解的数据会报错。

默认原始数组上限 8 GiB，可用 `--max-raw-gib` 调整；XML 文件上限 512 MiB。材料记录分块解压，表面提取仍需浮点场与网格工作内存。输入较大时先检查尺寸并选择合适的裁剪范围。

## 源码与验证

```text
run.py / run.cmd             命令行与 Windows 拖放入口
convert_vdata.py             兼容入口
examples/make_sample.py      纯标准库人工小样生成器
vdata_standalone/
  reader.py                  独立格式检查与材料记录解码
  planes.py                  平面字段解码与实验性界面场
  reconstruction.py          VOF/PLIC 引导的占比重建
  meshing.py                 表面提取、STL 写出和可选 QEM
  storage.py                 全部三角面保留的无损归档与分块读取
  planar.py                  共享顶点 MAT 与 Planar Solid 脚本
  cli.py                     流程、归档、坐标、材料表与 LSF
docs/tutorial.html           初学者完整图文教程
docs/assets/                教程图片和图示资源
tests/                      自造文件与几何的测试
requirements*.txt            基础、可选和测试依赖
pyproject.toml               包配置与 vdata-convert 命令
```

运行不依赖商业软件的合成测试：

```powershell
& $toolPython -m pip install -r requirements-dev.txt
& $toolPython -m pytest -q
```

v0.2.0 已有实现验证结果为 **123 个测试通过，另有 40 个子测试**。测试覆盖格式错误、坐标、几何和归档往返，并实际阻止厂商模块导入、安装目录访问和子进程启动，以检查独立路径。另有 7 个本机真实文件的读取验证，覆盖 um/nm、非零原点、单/多材料、含/不含界面字段；原始字节与独立解码公式相符，已知材料体积与历史量测相符。

这些验证不认证所有 Coventor 版本、原生界面等价性或 10 nm 光学位置精度。教程区分编码、几何、实际 CAD 导入、电磁求解和实验测量的验证层次。

## 作者本机 AH53 案例路径

下表仅解释教程和历史测量的本机案例，**其他使用者无需建立这些目录**。安装运行使用文首相对路径和自己的输入；真实输入与大型输出不随软件仓库分发。

| 本机路径 | 用途 |
|---|---|
| `D:\CoventorProjects\VDATA_Standalone` | 独立转换器开发目录 |
| `D:\CoventorProjects\AH53_Si_300nm_Nonideal\AH53_Si_300nm_Nonideal.vdata` | 已完成工艺仿真的 AH53，完整文件不含 PlaneEquations |
| `D:\CoventorProjects\AH53_FDTD_Alternatives\native_probe\native_crop_with_planes.vdata` | 含已保存平面的裁剪样例 |
| `D:\CoventorProjects\VDATA_Standalone_Validation` | 普通 Python 验证环境、测量与临时输出 |

方法依据：[scikit-image marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes)、[fast-simplification](https://pyvista.github.io/fast-simplification/)、[Garland–Heckbert QEM](https://www.mgarland.org/research/quadrics.html)。这些是公开算法与开源实现，不是 Coventor 命令。
