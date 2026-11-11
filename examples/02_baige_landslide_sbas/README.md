# 案例二：金沙江白格滑坡 SBAS 时序 InSAR

这是 InSAR Studio 的时序案例，目标是把「一叠 SLC → 干涉网络 → 解缠 → SBAS 反演 → 位移时间序列 → 速率图」完整走一遍，并且用公开的真实滑坡位移产品做对照，让读者知道每一步在做什么、结果能信到什么程度。

案例位于金沙江白格滑坡（西藏江达县与四川白玉县交界，约 `31.08°N, 98.70°E`）。2018 年 10 月和 11 月，白格发生两次大规模滑坡堵江，形成堰塞湖并溃决。本案例使用灾后 2018 年底至 2022 年的公开 InSAR 位移时间序列，重点不是复盘两次失稳，而是观察灾后坡体的持续变形。

> 和案例一一样，这是**真实公开产品派生**的回放栈，不是原始 SLC 重处理。数据包没有采集日期，本案例把 41 个历元映射到一条**重建的 6 天时间轴**上，只用于构建 SBAS 设计矩阵。绝对速率不能当作新的地质结论。

---

## 1. 案例事实

| 项目 | 内容 |
| --- | --- |
| 地点 | 金沙江白格滑坡，约 `31.08°N, 98.70°E` |
| 事件背景 | 2018-10-10、2018-11-03 两次滑坡堵江 |
| 数据产品 | 公开 InSAR 位移时间序列，199 个历元 |
| 源文件 | `disp_data.rar`，5.36 MiB |
| 源 MD5 | `0540704e40a1924c4a726aea6482855b` |
| 许可 | CC-BY-4.0 |
| DOI | [10.5281/zenodo.10605366](https://doi.org/10.5281/zenodo.10605366) |
| 工作台路线 | SBAS |
| 使用历元 | 41 个（每 5 个取 1 个，再补最后一个） |
| 重建时间轴 | `2018-12-14` 起，6 天间隔，最后 `2022-03-16` |
| 输出网格 | 51 × 95，像元约 `0.000185°`（约 20 m） |
| 参考点 | row 9, col 30（手动） |
| 峰值位移 | 工作台 `74.09 mm`，公开产品 `75.51 mm` |

源数据记录：

> Displacement time series of the Baige landslide. Zhiqiang Xiong. Zenodo. https://doi.org/10.5281/zenodo.10605366. License: CC-BY-4.0.

数据包只有 `Dt.mat`、`mask.mat`、`disp_par2` 三个文件，没有论文正文、采集日期、平台和轨道说明。`disp_par2` 给出 GAMMA 地理编码参数：`width=95`、`nlines=51`、角点 `98.69603°E, 31.08736°N`、像元 `1.8518519e-04°`。这些参数被原样写进回放栈。

---

## 2. 先看结果

- `results/figures/baige_velocity.png`：LOS 年均速率图。
- `results/figures/baige_timeseries.png`：峰值像元的时间序列对照，以及所有有效像元的散点。
- `results/summary.json`：完整指标。

关键数字：

| 指标 | 数值 |
| --- | --- |
| 有效像元 | 1800 / 4845（37.2%） |
| 速率中位数 | 8.24 mm/yr |
| 速率 5% / 95% 分位 | 0.17 / 14.16 mm/yr |
| 速率最大值 | 15.63 mm/yr |
| 末历元峰值（工作台） | 74.09 mm |
| 末历元峰值（公开） | 75.51 mm |
| 峰值差 | −1.42 mm |
| 全样本 RMSE | 0.204 mm |
| 95% 绝对残差 | 0.331 mm |
| 相关系数 | 0.99999 |

这些数字说明：在给定相位和参考点下，工作台的 SBAS 反演能忠实恢复公开位移时间序列。它不说明工作台能从原始 SLC 独立产出同等精度的产品，也不说明 8.24 mm/yr 是白格滑坡的真实长期速率——时间轴是重建的。

---

## 3. 目录里有什么

```text
examples/02_baige_landslide_sbas/
├── README.md                     ← 本教程
├── PROVENANCE.md                 ← 数据来源、变换、校验和
├── data/                         ← 已构建好的 41 历元回放栈（约 2.4 MB）
│   ├── slc.npy                   ← 41 × 51 × 95 complex64
│   ├── dem.npy                   ← 占位 DEM（全零）
│   ├── mask.npy                  ← 有效像元掩膜
│   ├── meta.json                 ← 工作台读取的元数据
│   ├── truth_disp_mm.npy         ← 公开位移时间序列，用作对照
│   └── replay_config.json        ← 构建参数与校验和
├── scripts/
│   ├── build_case.py             ← 下载 / 解压 / 读取 / 构建回放栈
│   └── run_case.py               ← 导入、跑 SBAS、验证、出图
└── results/
    ├── baige_disp_ts_mm.npy
    ├── baige_velocity_mm_yr.npy
    ├── baige_velocity_mm_yr.tif
    ├── baige_disp_last_mm.tif
    ├── summary.json
    └── figures/
```

`source/` 目录用于放下载的 RAR 和解压结果，不提交。

---

## 4. 路线 A：一条命令复现

### 4.1 准备

```bash
python -m pip install -r requirements.txt
python -m pip install h5py
python -m insar_studio doctor
```

`h5py` 只在重建数据时需要，因为 `Dt.mat` 是 MATLAB v7.3（HDF5）格式。`data/` 已经提交，直接跑结果不需要它。

### 4.2 重建回放栈

```bash
python examples/02_baige_landslide_sbas/scripts/build_case.py --download
```

脚本会：

1. 下载 `disp_data.rar` 到 `examples/02_baige_landslide_sbas/source/`；
2. 校验 MD5；
3. 用系统 `tar`（Windows 10+ 自带 bsdtar）解压；失败时尝试 `rarfile`；
4. 读取 `Dt.mat` 的 `(199, 95, 51)` 立方体，转成 `(199, 51, 95)`；
5. 读取 `mask.mat` 的 `(51, 95)` 掩膜；
6. 取索引 `0, 5, 10, …, 195, 198`，共 41 个历元；
7. 按 6 天间隔生成重建日期；
8. 用 C 波段波长把 mm 位移转成相位，写成复数 SLC；
9. 写出 `meta.json`、`replay_config.json` 和 `truth_disp_mm.npy`。

网络不稳时先手动下载，再指定文件：

```bash
python examples/02_baige_landslide_sbas/scripts/build_case.py --source D:\data\disp_data.rar
```

如果 RAR 已经在别处解压好，可以直接指向 `disp_data` 目录，跳过下载和解压：

```bash
python examples/02_baige_landslide_sbas/scripts/build_case.py --source-dir D:\data\disp_data
```

### 4.3 跑完整流程

```bash
python examples/02_baige_landslide_sbas/scripts/run_case.py
```

脚本创建临时工作区，导入 `data/`，跑 `network → interferometry → inversion`，把结果与 `truth_disp_mm.npy` 对比，写出 `results/`，最后删除临时工作区。保留工作区：

```bash
python examples/02_baige_landslide_sbas/scripts/run_case.py --keep
```

### 4.4 检查

```bash
python -c "import json;d=json.load(open('examples/02_baige_landslide_sbas/results/summary.json',encoding='utf-8'));print(d['validation_against_published_timeseries'])"
```

应看到 RMSE 约 `0.20 mm`、相关系数约 `0.99999`。

---

## 5. 路线 B：在 Web 界面里手动复现

### 5.1 启动

```bash
python -m insar_studio serve --open
```

或双击 `start.bat`。

### 5.2 新建 SBAS 项目并导入

1. 左侧 **“+ 新建”**。
2. 名称填 `白格滑坡 SBAS`。
3. 技术路线选 **SBAS-InSAR**。
4. 来源选 **“导入本地数据”**。
5. 数据路径填：

   ```text
   D:\opencode_workspace\insar-studio\examples\02_baige_landslide_sbas\data
   ```

6. 勾选 **“导入的 SLC 已经配准”**。
7. 创建项目。

进入 **“数据”** 页核对：

- 景数：41；
- 尺寸：51 × 95；
- 日期：`2018-12-14` 到 `2022-03-16`；
- 日期来源会显示为推断/重建（`dates_source: reconstructed`）；
- 掩膜：1800 个有效像元。

**看到日期是重建的，不要惊讶。** 源数据包没有采集日期，这是本案例最需要记住的限制。

### 5.3 生成网络

进入 **“网络”** 页：

1. 最大时间基线填 `60` 天；
2. 最大干涉对填 `300`；
3. 生成网络。

41 个历元、30 天间隔、60 天阈值，每个历元连前后各 2 个，实际生成 79 对。网络应当连通，没有孤立日期。

### 5.4 干涉参数

进入 **“干涉”** 页，专家模式：

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `multilook` | 1 | 网格只有 51 × 95，再降会丢滑坡边界 |
| `filter` | `none` | 回放相位来自已处理的位移产品，不需要再滤波 |
| `coherence_window` | 5 | 小窗口适应小影像 |
| `min_coherence` | 0.0 | 回放栈相干性接近 1 |
| `unwrap_coherence_min` | 0.0 | 不裁掉有效像元 |
| `unwrap_method` | `quality` | 内置质量引导解缠 |
| `remove_ramp` | 关闭 | 滑坡位移是真实长波信号 |

运行干涉。79 对在小影像上很快，通常几秒到十几秒。

### 5.5 大气校正

本案例跳过大气校正：

- 源数据已经是位移产品，大气延迟已经包含在公开处理里；
- 再做一次经验高程校正或时间高通，会重复扣除真实信号；
- 教学重点是 SBAS 链路，不是大气方法对比。

在 **“大气”** 页选择 **“不校正”**，或者直接进入反演。

### 5.6 反演参数

进入 **“反演”** 页：

| 参数 | 值 | 原因 |
| --- | --- | --- |
| 正则权重 `regularization` | `0.0` | 数据接近无噪声，正则化会压缩峰值 |
| 稳健迭代 `robust_iterations` | `1` | 回放数据没有明显离群对，1 次足够 |
| 估计 DEM 误差 | 关闭 | DEM 是占位的全零，估计没有意义 |
| APS 方法 | 关闭 | 见上一节 |
| 空间参考 | 手动指定 | 避免自动策略选到滑坡体 |
| 参考行 `row` | `9` | 稳定远场像元 |
| 参考列 `col` | `30` | 稳定远场像元 |

运行反演。

### 5.7 看成果

进入 **“成果”** 页：

- 速率图单位 mm/yr；
- 时间序列面板可以点选像元；
- 峰值像元在 `row 39, col 44`；
- 参考像元 `row 9, col 30` 的位移应始终为 0。

把峰值像元的时间序列和 `results/figures/baige_timeseries.png` 对照，应该看到工作台曲线略低于公开曲线，但形态一致，末历元差约 1.4 mm。

### 5.8 导出

成果页可导出：

- 速率 GeoTIFF；
- 位移时间序列数组；
- 逐历元位移图。

导出的速率是**基于重建时间轴**的速率。引用前请读第 7 节。

---

## 6. 参数为什么这样定

### 6.1 为什么只取 41 个历元

原始 199 个历元如果全用：

- 网络对数和反演矩阵会大很多；
- 教学演示不需要 199 个几乎共线的观测；
- 每 5 个取 1 个后仍有 41 个历元、79 对，足以稳定估计线性速率和季节项。

补最后一个历元是为了让时间序列结束在 2022-03-16，而不是 2022-02-26，保证末历元峰值和公开产品对齐。

### 6.2 为什么重建 6 天时间轴

SBAS 设计矩阵需要每个历元的时间。源包没有日期，只能用间隔假设。6 天是 Sentinel-1 的典型重访周期，与 199 个历元、约 3.3 年的跨度大致自洽。

这条轴只影响：

- 速率的绝对尺度（mm/yr）；
- 季节项的频率解释。

它不影响：

- 每个历元的位移拟合；
- 位移时间序列的形态；
- 峰值相对大小。

所以本案例的位移和 RMSE 可信，绝对速率只能作相对比较。

### 6.3 为什么参考点选 (9, 30)

`(9, 30)` 是掩膜内末历元位移为 0 的像元，位于滑坡边界外的稳定区。用它做参考，滑坡体的位移才是“相对于稳定地面”的位移。

自动参考点会选相干性最高、时序最稳定的像元，但回放栈相干性几乎处处为 1，自动策略可能落在滑坡体内部，把一部分真实位移当成基准扣掉。手动指定避免这个问题。

### 6.4 为什么正则化为 0

正则化会在大矩阵求解中压制病态方向，代价是峰值被缩小。回放数据的相位由位移直接生成，噪声极低，正则化只会让结果偏离真值。实测：

- `regularization=0.0`：RMSE 0.20 mm，峰值 74.1 mm；
- `regularization=0.01`：RMSE 2.36 mm，峰值 15.0 mm；
- `regularization=0.3`：RMSE 1.47 mm，峰值 17.7 mm。

所以本案例明确关闭正则化。真实含噪数据不应照搬这个设置。

### 6.5 为什么不做大气校正

源数据是已经处理好的位移产品。再跑经验高程校正，会把与地形相关的真实滑坡位移当成大气扣掉；再跑时间高通，会把长期趋势当成 APS 扣掉。教学案例选择“不校正”，并在文档里说明原因，而不是假装做了一次大气处理。

---

## 7. 结果怎么看

### 7.1 速率图

- 滑坡体中部有一条北西—南东向的高速率带，峰值约 15.6 mm/yr；
- 向边界逐渐降到 0；
- 掩膜外为空白；
- 负值很小（最小约 −0.14 mm/yr），说明参考点选得合适。

这个形态与白格滑坡灾后持续变形的公开认识一致，但**速率数值依赖重建时间轴**，不能直接写进论文当绝对速率。

### 7.2 时间序列

峰值像元 `(39, 44)` 的公开序列：

```text
0 → 6.9 → 12.3 → 16.7 → 21.8 → 25.2 → 28.9 → 39.2 → 75.5 mm
```

工作台序列：

```text
0 → 6.9 → 12.2 → 16.5 → 21.7 → 25.0 → 28.7 → 39.0 → 74.1 mm
```

两条曲线几乎重合，末历元差 1.4 mm。差异来自解缠数值路径和最小二乘拟合，不是时间轴。

### 7.3 哪些结论能说

能说：

- 工作台 SBAS 链路能在该数据上恢复公开位移时间序列；
- 滑坡体存在持续的正 LOS 形变，空间形态与公开产品一致；
- 参考点、网络和参数设置是可复现的。

不能说：

- 白格滑坡的绝对速率是 8.24 mm/yr；
- 2022 年 3 月之后滑坡会如何演化；
- 本案例能替代原始 SLC 的时序处理。

### 7.4 为什么峰值比公开低 1.4 mm

可能原因：

1. 解缠相位在峰值附近有轻微平滑；
2. SBAS 最小二乘对所有对加权，峰值被轻微拉低；
3. 参考点与公开产品的基准不完全相同；
4. 浮点相位转换的舍入。

1.4 mm 相对 75.5 mm 约 1.9%，对教学案例可以接受。如果要做毫米级滑坡监测，应该用原始 SLC 和成熟时序软件交叉验证。

---

## 8. 如果要从原始 SLC 开始

需要：

1. 确定白格滑坡覆盖的 Sentinel-1 轨道（升轨/降轨）和帧号；
2. 从 Copernicus Data Space Ecosystem 或 ASF 下载 2018-12 至 2022-03 的 SLC 和精密轨道；
3. 用 SNAP / ISCE2 / Gamma 做 TOPS 配准、干涉、去地形、解缠；
4. 导出配准后的复数栈或解缠相位序列；
5. 导入 InSAR Studio，按本案例的参数跑 SBAS。

这条路径的数据量是几十到几百 GB，远超教学仓库。本案例把“拿到位移产品之后怎么验证工作台”讲清楚，原始 SLC 处理请使用成熟生产链。

---

## 9. 常见问题

**Q：为什么日期是 2018-12-14 到 2022-03-16，源数据不是没有日期吗？**

这是重建的 6 天时间轴，不是真实采集日期。`meta.json` 里 `dates_source: reconstructed`，教程和结果都标注了。它只用于 SBAS 设计矩阵。

**Q：为什么 `data/dem.npy` 全是 0？**

回放相位已经包含位移，DEM 只用于导入器合法性检查。全零 DEM 不参与物理计算。

**Q：为什么速率中位数是 8.24 mm/yr，而不是峰值 15.6？**

中位数覆盖整个掩膜，包括边界稳定区；峰值只出现在滑坡中部。看速率图比看单一数字更有意义。

**Q：为什么正则化关闭后结果反而更贴近公开产品？**

因为回放数据几乎没有噪声，正则化的收缩偏差占主导。真实数据要重新评估。

**Q：可以把 41 个历元全部换成 199 个吗？**

可以。修改 `build_case.py` 的 `EPOCH_STEP=1`，重建数据，网络和反演会更慢，但结果应更平滑。教学案例选 41 是为了体积和速度。

**Q：为什么掩膜只有 37%？**

源数据只对滑坡及周边有效区给出位移，其余像元为 0。`mask.mat` 保留了 1800 个有效像元，回放栈原样使用。

**Q：可以把这个案例写进论文吗？**

可以引用数据源和本仓库作为教学复现；绝对速率必须标注“重建时间轴”，并引用原始数据论文。科研使用请用原始 SLC 和成熟时序软件独立验证。

---

## 10. 引用与许可

数据：

> Xiong, Z. Displacement time series of the Baige landslide. Zenodo. https://doi.org/10.5281/zenodo.10605366. License: CC-BY-4.0.

背景文献（建议一并引用）：

> Xiong, Z., Feng, G., Feng, Z., Miao, L., Wang, Y., Yang, D., & Luo, S. (2020). Pre- and post-failure spatial-temporal deformation pattern of the Baige landslide retrieved from multiple radar and optical satellite images. Engineering Geology, 279, 105880. https://doi.org/10.1016/j.enggeo.2020.105880

本案例代码和文档随 InSAR Studio 以 MIT 许可发布；`data/` 中的派生数组沿用源数据的 CC-BY-4.0，分发时请保留署名、DOI 和“重建时间轴”说明。

变换细节、文件校验和和逐步命令见 [PROVENANCE.md](PROVENANCE.md)。
