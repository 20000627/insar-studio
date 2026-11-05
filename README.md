# InSAR Studio · 时序干涉形变处理平台

一个完全本地运行的 InSAR 时序形变处理工作台。它把「数据接入 → 配准 → 干涉网络 → 差分干涉 → 大气校正 → 时序反演 → 成果输出」串成一条可复现的处理链，并提供中文分步向导。零基础用户可以用默认参数一键跑完，进阶用户可以在专家模式里调每一个关键参数。

支持四条技术路线：

| 路线 | 适用场景 | 默认网络 |
| --- | --- | --- |
| **SBAS-InSAR** | 面状沉降、滑坡、城市形变，兼顾精度与覆盖 | 小基线集，最大时间基线 72 天 |
| **PS-InSAR** | 城市建筑、岩石裸露区，毫米级点目标监测 | 单主影像星形 + 相邻对闭环 |
| **DS-InSAR** | 植被、裸土等低相干区 | 相位链接 + 分布式散射体联合解算 |
| **D-InSAR** | 同震、滑坡、单次事件的快速差分 | 单干涉对（可选日期） |

> 本项目是本地科研/教学与算法验证工具，不是经过计量认证的生产系统。真实业务请使用经严格验证的 SNAP / ISCE / Gamma / StaMPS 等生产链，并遵守数据许可与安全规范。

![成果页图件](docs/images/ui-results.png)

**文档**：[用户手册](docs/用户手册.md) · [使用教程](docs/使用指南.md) · [技术说明](docs/技术说明.md)

## 目录

- [快速开始](#快速开始)
- [四套演示与精度](#四套演示与精度)
- [七步工作流](#七步工作流)
- [数据接入](#数据接入)
- [技术路线与关键算法](#技术路线与关键算法)
- [大气校正](#大气校正)
- [默认参数](#默认参数)
- [成果文件](#成果文件)
- [项目目录结构](#项目目录结构)
- [命令行](#命令行)
- [测试](#测试)
- [常见问题](#常见问题)
- [限制与声明](#限制与声明)

## 快速开始

### 方式一：Windows 一键启动（推荐）

1. 确认已安装 **Python 3.9 – 3.11**（安装时勾选 “Add python.exe to PATH”）。
2. 双击项目根目录的 `start.bat`。
   - `scripts/bootstrap.py` 会检查运行依赖；缺失时自动创建本地 `.venv` 并安装 `requirements.txt`（只需一次）。
   - 默认从 `8765` 起自动选择空闲端口，并自动打开浏览器；终端会打印实际地址。
   - 端口被占用不会导致启动失败，会自动顺延到下一个空闲端口。固定端口、禁用浏览器等参数见[命令行](#命令行)。
3. 在左侧「项目库」选择任意演示项目，或点击「+ 新建」创建新项目。
4. 进入「成果」步骤，点击「一键跑完全流程」，等待任务控制台显示「已完成」。

首次安装依赖需要联网；之后完全离线可用。服务只监听 `127.0.0.1`，不会把数据暴露到局域网。

### 方式二：手动启动

```powershell
cd D:\path\to\insar-studio
python -m pip install -r requirements.txt
python -m insar_studio serve --open
```

默认从 `8765` 起自动选择空闲端口并打开浏览器；终端会打印实际地址。按 `Ctrl+C` 停止服务。需要固定端口时加 `--port 8765 --strict-port`；`--port 0` 表示交给操作系统分配。

### 第一次使用建议

1. 打开 `SBAS 演示` 项目，先看「成果」页的速率图、相干性图和时间序列。
2. 点「一键跑完全流程」，观察任务控制台和自动推进的步骤。
3. 新建一个演示项目，切换到 PS / DS / D-InSAR，比较不同路线的结果。
4. 再尝试导入自己的数据（见[数据接入](#数据接入)）。

## 四套演示与精度

演示数据是确定性合成的 Sentinel-1 风格 24 景时序栈（220 × 260 像元，2023-01-03 至 2023-10-06，12 天重访），包含已知形变、分层大气、DEM 误差、轨道 ramp、亚像元配准偏移和相干性变化。因此可以用真值定量检查整条链路，而不仅仅是“看起来能跑”。

当前版本对本机四套演示的回归结果：

| 路线 | 位移 RMSE | 速率 RMSE | 速率空间相关 | 综合评价 | 备注 |
| --- | ---: | ---: | ---: | --- | --- |
| SBAS | **2.01 mm** | **3.12 mm/yr** | 0.957 | 优秀 | 80 对，有效像元 91.0% |
| PS | **2.30 mm** | **4.16 mm/yr** | 0.912 | 优秀 | 45 对，42,326 个 PS 点 |
| DS | **2.48 mm** | **3.22 mm/yr** | 0.894 | 优秀 | 80 对，分布式散射体联合解算 |
| D-InSAR | **1.46 mm** | — | — | 优秀 | 单对，位移偏差 0.05 mm |

本机最近一次全流程回归耗时：D-InSAR 6.1 s、DS 33.2 s、PS 53.6 s、SBAS 27.0 s；四套项目体检均为 0 错误 / 0 警告。耗时随机器性能变化，仅作量级参考。

合成数据只能验证算法链路与数值实现，**不代表真实场景精度**。真实数据的精度取决于影像质量、轨道、DEM、配准、解缠和大气校正质量。

![大气与网络诊断](docs/images/ui-atmosphere-network.png)

## 七步工作流

界面顶部是固定的七步导航，每一步都有「引导」说明；右侧「专家」开关会展开更多参数。

### 任务控制台与结果一致性

- 处理任务在后台线程执行，同一项目同时只允许一个任务；重复提交会被拒绝（HTTP 409），界面会提示已有任务在运行。
- 任务运行中可以「取消任务」；刷新页面后若任务仍在运行，页面会自动重连并继续显示进度。
- 修改参数或切换技术路线不会打断正在运行的任务；路线切换在任务内部生效，并重新套用该路线的默认采样参数。
- 重跑任一阶段前会删除该阶段的下游产品，避免新旧结果混在一套成果里；项目体检会把参数已变化的阶段标记为「参数待重跑」。
- 顶部「项目体检」是只读检查，覆盖数据、参数指纹、配准、网络、干涉、大气、反演、报告与磁盘空间；任务运行期间返回 409。

### 1. 项目

- 新建演示项目：选择技术路线，自动生成 24 景合成数据。
- 新建导入项目：先建空项目，再到「数据」步骤导入本地栈。
- 项目库支持搜索、切换、删除；每个项目独立保存在 `workspace/projects/<id>/`。
- 顶部显示数据源、时间范围、影像数、尺寸和已完成步骤。

### 2. 数据

- 演示项目直接显示振幅预览、DEM、有效掩膜和真值信息。
- 导入项目支持本地 SLC 栈、DEM、GACOS 目录。
- Sentinel-1 下载面板支持 Copernicus Data Space 与 ASF 的搜索/下载（实验性，见下文）。

### 3. 网络

- 自动检查 SLC 是否已配准；没有配准时会先运行配准。
- 按技术路线生成干涉对：SBAS 小基线、PS 星形+相邻、DS 小基线、D-InSAR 单对。
- 显示时间基线/垂直基线散点图、连通分量、冗余度和干涉对列表。
- 可手动修改最大时间基线、最大垂直基线、最小相干性等参数后重建网络。

### 4. 干涉

- 逐对生成干涉相位、相干性、滤波相位、缠绕相位和解缠相位。
- 支持相位滤波：Boxcar、Goldstein、Gaussian、NL-Means、关闭。
- 支持解缠：内置质量引导、SNAPHU（可选）、skimage 最小二乘。
- 可逐对浏览振幅、缠绕相位、滤波相位、解缠相位、相干性和残差图。

### 5. 大气

- **GACOS**：演示/本地 `.ztd`/`.tif`/`.npy`/自定义在线服务。
- **经验高程校正**：相位—高程稳健回归，不需要外部数据。
- **不校正**：保留原始相位，由反演阶段的时间高通压掉部分 APS。
- 诊断图显示逐日 APS 屏、校正前后相位分布和逐对离散度。

### 6. 反演

- SBAS/PS/DS：加权最小二乘 + 正则化 + 稳健迭代，解算时序位移和速率。
- D-InSAR：单对差分相位直接转位移。
- 可选 DEM 误差联合估计、去线性/二次 ramp、参考像元模式。
- 残差、时序相干性、DEM 误差和 APS 屏都会输出为图件和数组。

### 7. 成果

- 速率/位移主图、相干性、残差、DEM 误差、大气诊断、网络图、配准增益图。
- 任意像元时间序列查询与 CSV 导出，支持跳到形变峰值和参考点。
- 真值校验表（仅演示数据）。
- 点位表 CSV、Markdown + DOCX 报告、ZIP 成果包下载。
- 所有图件可点击放大查看。

## 数据接入

### 推荐的真实数据流程

InSAR Studio 负责**时序处理与反演**，不负责 Sentinel-1 的 burst 拼接、精密轨道、TOPS 配准和地理编码。真实数据建议先在 SNAP / ISCE / Gamma 中完成：

1. 轨道校正、burst 拼接、去斜、TOPS 配准；
2. 生成配准后的复数 SLC 栈；
3. 提供 DEM 与有效掩膜；
4. 导出为 GeoTIFF / ENVI / NumPy 栈，再导入本工具。

本工具不直接导入干涉图栈、解缠相位或现成的形变产品，也不支持 GRD、RAW、光学影像、点云和 GNSS。干涉图和时序产品必须由 SLC 栈在本工具或其他 InSAR 软件中生成。完整的影像类型、处理原理和适用边界见[使用教程第 5 章](docs/使用指南.md#5-支持的影像类型与处理原理)。

### 支持格式

| 类型 | 格式 |
| --- | --- |
| SLC 栈 | 目录 + `stack.json`/`meta.json` + `slc.npy`；单个 `.npy`；单个 `.npz` 打包；ENVI `.hdr/.dat`；复数 GeoTIFF 目录；Sentinel-1 SAFE（实验性） |
| DEM | `.tif/.tiff`、`.npy`、ENVI |
| GACOS | `.ztd + .rsc`、`.tif`、`.npy` |
| 不支持 | 干涉图、解缠相位、形变产品、GRD、RAW、光学影像、点云、GNSS |

实数振幅/强度图可以导入，但会被转成零相位复数，只能做界面和流程演示，不能产生有意义的形变结果。

NumPy 栈形状为 `(n_dates, rows, cols)`，复数类型 `complex64`。最小示例：

```text
my_stack/
  slc.npy          # complex64, (n_dates, rows, cols)
  dem.npy          # float32, (rows, cols)，可选
  mask.npy         # bool, (rows, cols)，可选
  meta.json        # 日期、基线、平台等元数据
```

`meta.json` 示例：

```json
{
  "dates": ["2023-01-03", "2023-01-15", "2023-01-27"],
  "platform": "Sentinel-1A",
  "orbit_direction": "DESCENDING",
  "relative_orbit": 42,
  "bperp_m": [-35.2, 12.8, 64.1],
  "incidence_deg": 38.5,
  "range_spacing_m": 2.33,
  "azimuth_spacing_m": 13.97
}
```

如果缺少 `bperp_m`，工具会把垂直基线置零并关闭 DEM 误差估计；如果缺少 `dates`，会按 12 天间隔补齐。导入后建议先在「数据」页核对日期、尺寸和掩膜。

也可以把同一组数组打成一个 `.npz`：

```text
stack.npz
  slc     # complex64, (n_dates, rows, cols)，必需
  dem     # float32, (rows, cols)，可选
  mask    # bool, (rows, cols)，可选
  dates   # 字符串数组，可选
  meta    # JSON 字符串，可选
```

`dates` 优先从 `dates` 数组读取，其次从 `meta["dates"]` 读取；缺失日期按 12 天间隔补齐。导入时按日期升序重排影像，并同步重映射 `bperp_m` 与 `ref_index`，不需要手动调整参考景。

### Sentinel-1 下载（实验性）

「数据」步骤内置两个下载适配器：

- **Copernicus Data Space**：OData 搜索 + `$value` 下载，需要 CDSE 邮箱和密码。
- **ASF DAAC**：公开搜索，SLC 下载需要 Earthdata 账号。

凭据只用于当前请求，不写入 `project.json`。大文件下载容易受网络和会话过期影响，生产环境建议使用官方客户端或 `aria2`，下载后再导入。

## 技术路线与关键算法

### 配准

- 以第一景为参考，使用相位相关估计整像元偏移，再用局部相关做亚像元精化。
- 输出配准后的 SLC 栈、逐景偏移量、配准残差和相干性增益图。
- 真实 TOPS 数据仍需在 SNAP/ISCE 中完成 burst 级配准，本工具只处理已配准或近似配准的栈。

### 干涉

- 复共轭相乘生成干涉相位，按窗口估计相干性。
- 可选多视、相位滤波、去平地/去地形相位（依赖 DEM 和基线）。
- 解缠前按相干性和掩膜剔除低质量像元。

### 解缠

- 默认质量引导区域生长，适合中等噪声和断点较多的场景。
- SNAPHU 可选；未安装时自动回退内置算法，不会中断流程。
- skimage 最小二乘解缠适合连续、低噪声相位。

### 反演

SBAS/PS/DS 使用联合观测方程：

```text
Δφ(i,j) = (4π/λ) · [d(t_j) - d(t_i)] + (4π/λ) · B⊥(i,j) · h_error / (R·sinθ) + APS + noise
```

对每个像元做加权最小二乘，加入一阶/二阶时间正则和稳健重加权，抑制解缠误差和离群干涉对。PS 路线额外做振幅离差/相干性选点；DS 路线先做相位链接和自适应多视，再联合解算。

D-InSAR 直接由单对解缠相位转 LOS 位移：

```text
d_LOS = -λ / (4π) · Δφ_unwrapped
```

所有时序结果统一参考到一个稳定像元；演示数据的真值校验也减去同一像元的历史，避免参考基准造成的假误差。

### 参考像元

- `auto`：自动选择时序相干性高、残差小的稳定区域。
- 也可在专家参数中指定行列，或在成果页用「参考点」按钮查询。

## 大气校正

### GACOS

- **演示**：合成逐日大气屏，与演示数据的大气真值一致，用于验证校正上限。
- **本地**：读取目录中的 `.ztd/.rsc`、`.tif`、`.npy`，按日期匹配，重采样到干涉图网格，转 LOS 延迟后逐对差分。
- **在线**：可配置服务地址、邮箱和令牌；不同 GACOS 部署的接口不一致，因此没有硬编码官方端点。

### 经验高程校正

对每个干涉对做相位—高程稳健回归（可选一阶/二阶），扣除分层大气。适合没有 GACOS 但研究区地形起伏明显的场景。校正过度会吃掉真实长波形变，建议对比校正前后的速率图和残差。

### 时间高通 APS

反演阶段可对残余 APS 做时间高通/空间低通分离（`aps_iterations=1`，`aps_sigma_px=10`）。这对短周期大气扰动有效，但不能替代外部天气模型。

## 默认参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `multilook` | 2 | 多视因子；PS 默认 1 |
| `coherence_window` | 5 | 相干性估计窗口 |
| `filter` | boxcar | 相位滤波方法 |
| `boxcar_size` | 3 | Boxcar 窗口 |
| `goldstein_alpha` | 0.65 | Goldstein 滤波强度 |
| `unwrap_method` | quality | 质量引导解缠 |
| `unwrap_coherence_min` | 0.18 | 解缠相干性下限 |
| `max_temporal_baseline` | 72 天 | SBAS/DS 最大时间基线 |
| `max_perpendicular_baseline` | 180 m | 最大垂直基线 |
| `min_coherence` | 0.25 | 反演有效像元相干性下限 |
| `estimate_dem_error` | false | 是否联合估计 DEM 误差 |
| `remove_ramp` | true | 解缠后去单平面线性趋势（当前实现不含二次项） |
| `regularization` | 0.30 | 时间正则强度 |
| `aps_iterations` | 1 | 残余 APS 迭代次数 |
| `aps_sigma_px` | 10 | APS 空间尺度 |
| `reference_mode` | auto | 参考像元选择 |
| `ps_adi_max` | 0.28 | PS 振幅离差阈值 |
| `ps_coherence_min` | 0.62 | PS 相干性阈值 |
| `ds_patch` | 9 | DS 同质块大小 |
| `ds_overlap` | 0.5 | DS 块重叠率（当前实现未使用） |

专家模式还会暴露 Goldstein 块大小、高斯 σ、稳健迭代次数、DEM 误差开关、GACOS 服务地址/令牌等。

## 成果文件

每个项目在 `workspace/projects/<id>/products/` 下输出：

```text
products/
  arrays/          # disp_ts_mm.npy, velocity_mm_yr.npy, temporal_coherence.npy,
                   # residual_mm.npy, dem_error_m.npy, aps_screens.npy, valid.npy
  geotiff/         # disp_YYYYMMDD.tif, velocity_mm_yr.tif, temporal_coherence.tif, dem_error_m.tif
  csv/             # timeseries_points.csv
  figures/         # velocity.png, temporal_coherence.png, residual.png, dem_error.png,
                   # atmosphere.png, network.png, coreg_coherence_gain.png
  report/          # report.md, report.docx
  exports/         # ZIP 成果包
  inversion.json   # 反演摘要
  validation.json  # 真值校验（演示数据）
```

ZIP 成果包包含项目配置、产品数组、GeoTIFF、CSV、图件和报告，便于归档或交付。

## 项目目录结构

```text
insar-studio/
  insar_studio/
    api.py            # Flask API 与页面路由
    cli.py            # serve / demo / run / doctor / projects
    pipeline.py       # 五阶段处理编排
    project.py        # 项目持久化
    validation.py     # 合成真值校验
    report.py         # Markdown/DOCX 报告与 ZIP
    core/             # 配准、网络、干涉、滤波、解缠、大气、反演、PS/DS
    static/           # 前端 CSS/JS
    templates/        # 单页应用模板
  tests/              # pytest 测试
  docs/               # 用户手册、算法说明、截图
  scripts/            # 一键启动引导 bootstrap.py
  workspace/projects/ # 项目与演示数据
  start.bat / start.sh # Windows / Linux、macOS 一键启动
  requirements.txt
  pyproject.toml
```

## 命令行

```powershell
# 启动 Web 界面（自动选择 8765 起的空闲端口，不自动开浏览器）
python -m insar_studio serve

# 启动并自动打开浏览器
python -m insar_studio serve --open

# 固定端口：被占用时直接退出；--port 0 交给系统分配
python -m insar_studio serve --port 8765 --strict-port
python -m insar_studio serve --port 0

# 创建并运行一个演示项目（sbas / ps / ds / dinsar）
python -m insar_studio demo --name "我的 SBAS 演示" --technique sbas

# 只生成演示数据，不处理
python -m insar_studio demo --name "空演示" --technique ps --no-run

# 重跑已有项目
python -m insar_studio run prj_xxxxxxxx_xxxxxx

# 环境自检
python -m insar_studio doctor

# 列出项目
python -m insar_studio projects
```

`run` 会覆盖已有产品；演示数据会保留。批量回归时可以组合 `demo` + `run`。

## 测试

```powershell
python -m pytest -q
```

当前测试覆盖：

- 网络构建（SBAS 连通性、PS 星形、D-InSAR 单对）；
- 滤波与质量引导解缠；
- SNAPHU 缺失回退；
- 项目管理、参数继承、路径越界防护、阶段下游清理与上游图件保留；
- Windows 下产物删除的重试、并发重写与持续占用报错；
- 运行中产物被清理时的目录扫描容错；
- 本地 NumPy 栈与 `.npz` 导入、日期排序与参考景重映射；
- D-InSAR 全流程、真值校验、报告与 ZIP 导出；
- Flask API 项目生命周期、404/400 处理、资源路径安全、任务互斥与路线切换。

当前为 **30 个测试**，全量约 15 秒（不含四套演示的全流程回归）。

## 常见问题

**端口被占用**

默认会自动顺延到下一个空闲端口，终端会打印实际地址，无需处理。若希望固定端口并在被占用时明确失败：

```powershell
python -m insar_studio serve --port 8765 --strict-port
```

`--port 0` 表示由操作系统分配端口。也可以结束占用 8765 的进程后重试。

**为什么项目体检提示「参数待重跑」**

说明上次处理后参数被修改过，当前成果仍对应旧参数。按体检给出的下一步重跑该阶段即可；重跑会清理下游产品。体检本身是只读的，不会修改数据。

**任务运行时为什么不能改参数或重复提交**

同一项目同一时间只允许一个处理任务，避免两个任务同时写 `products/` 造成产品混用。等待当前任务完成或取消后再提交；刷新页面后会自动重连到正在运行的任务。

**依赖安装失败**

优先使用 Python 3.9–3.11 的 64 位环境。`scikit-image`、`scipy` 需要与 NumPy 版本匹配；本项目在 `numpy 1.24 / scipy 1.10 / scikit-image 0.20 / matplotlib 3.7` 上验证。若使用 conda：

```powershell
conda create -n insar python=3.11 numpy=1.24 scipy=1.10 scikit-image=0.20 matplotlib=3.7 flask=2.2 -c conda-forge
conda activate insar
pip install -r requirements.txt
```

**为什么没有 SNAPHU 也能跑**

内置质量引导解缠是默认算法，SNAPHU 只是可选加速/对比方案；未安装时会自动回退，不会中断。

**真实数据跑出来全是噪声**

先检查：SLC 是否已配准、轨道/基线元数据是否正确、DEM 是否同网格、掩膜是否覆盖有效区域、相干性阈值是否过低。真实 TOPS 数据不要直接拿未拼接的 burst 导入。

**速率图方向反了**

本工具约定 LOS 正值朝向卫星（抬升/靠近卫星为正）。不同软件的符号约定可能相反，比较时先确认参考像元和符号。

**中文显示为乱码**

Web 界面使用 UTF-8，不受影响。命令行在 GBK 控制台下个别符号可能显示异常；建议使用 `start.bat`（会切换 UTF-8）或 Windows Terminal。

**成果包太大**

ZIP 包含全部干涉图和数组。只需要交付图件时，可直接从 `products/figures` 和 `products/report` 取文件。

## 限制与声明

- 合成演示用于算法回归，不能外推为真实精度指标。
- 真实 Sentinel-1 处理仍依赖外部 TOPS 配准、精密轨道和地理编码；本工具不替代 SNAP/ISCE/Gamma。
- 不包含大气数值天气预报下载、GNSS 融合、角反射器识别、断层反演等生产级功能。
- 不包含用户认证与多用户隔离；服务默认只监听本机，请勿直接暴露到公网。
- 所有数据与产品保存在本地 `workspace/`，删除项目会删除对应目录，请自行备份。

---

**版本**：1.2.0 · **Python**：3.9+ · **平台**：Windows / Linux / macOS（Web UI 本地运行）
