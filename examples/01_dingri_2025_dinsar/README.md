# 案例一：2025-01-07 西藏定日 Mw 7.0 地震 D-InSAR

这是 InSAR Studio 的第一个真实案例，目标只有一条：**让第一次接触 InSAR 的人，从下载数据开始，在自己的电脑上得到一张同震 LOS 形变图，并且知道这张图是怎么来的、哪里能信、哪里不能信。**

案例使用 2025 年 1 月 7 日西藏定日地震的公开升轨解缠相位产品，构造一个工作台可以直接读取的两景复数 SLC 回放栈，然后完整走一遍 D-InSAR 路线。案例目录里已经放好回放栈、结果、脚本和教程；你可以直接打开结果，也可以从头重建一遍。

> 这是一个**真实公开产品派生**的教学案例，不是原始 Sentinel-1 SLC 重处理。回放栈的相位来自公开解缠产品，工作台只负责后续的相位到位移转换、参考点、出图和验证。原始 SLC 的获取与 SNAP / ISCE / Gamma 预处理路径在文末单独说明。

---

## 1. 案例事实

| 项目 | 内容 |
| --- | --- |
| 事件 | 2025-01-07 西藏定日地震，数据集标注 Mw 7.0（部分机构给 Mw 7.1） |
| 构造背景 | 藏南裂谷系，正断层型事件 |
| 卫星 / 轨道 | Sentinel-1A，升轨 |
| 影像日期 | 2025-01-05（震前）与 2025-01-17（震后），间隔 12 天 |
| 源产品 | GAMMA `adf_quad.unw.geo.tif` 解缠相位，WGS 84 地理网格 |
| 源文件 | `ASC_20250105_20250117.adf_quad.unw.geo.tif`，26.27 MiB |
| 源 MD5 | `894d51cb912ff24ece4af8931d8ee57c` |
| 许可 | CC-BY-4.0 |
| DOI | [10.5281/zenodo.14836721](https://doi.org/10.5281/zenodo.14836721) |
| 工作台路线 | D-InSAR |
| 输出网格 | 500 × 500，像元 0.0025° |
| 参考点 | row 24, col 63（手动） |
| 相位压缩比 | 10（工作台输出 ×10 恢复公开 LOS） |

源数据是论文配套产品：

> Geodetic Observations and Seismogenic Structures of the 2025 Mw 7.0 Dingri Earthquake: The Largest Normal Faulting Event in the Southern Tibet Rift. Zenodo record 14836721.

---

## 2. 先看结果

打开下面两张图，不需要安装任何东西：

- `results/figures/dingri_los_mm.png`：工作台输出的同震 LOS 形变图（已 ×10）。
- `results/figures/dingri_validation.png`：与公开产品逐像元对比，以及去掉常数基准差后的残差图。

数值结果在 `results/summary.json`：

| 指标 | 数值 |
| --- | --- |
| LOS 最小值 | −323.88 mm |
| LOS 最大值 | 1205.16 mm |
| LOS 5% / 95% 分位 | −134.75 / 182.55 mm |
| 与公开产品相关系数 | 1.0000 |
| 去常数基准差后的 RMSE | 约 0.00001 mm |
| 95% 绝对残差 | 约 0.00002 mm |
| 有效像元比例 | 98.4% |

这么大的相关系数和极小的残差不是“算法精度”，而是因为回放栈的相位本来就来自这份公开产品。它证明的是：**工作台的 D-InSAR 链路能把输入的相位忠实转换成 LOS 位移，参考点和比例没有搞错。** 它不能证明工作台能替代 SNAP / ISCE / Gamma 的原始 SLC 处理。

真正有信息量的是形变形态本身：震中附近一条近南北向的强正 LOS 形变带，峰值超过 1.2 m；西侧出现负值瓣。升轨几何下，LOS 正负对应地表向卫星靠近或远离，不能直接读成“上升/下降”。

---

## 3. 目录里有什么

```text
examples/01_dingri_2025_dinsar/
├── README.md                     ← 本教程
├── PROVENANCE.md                 ← 数据来源、变换、校验和
├── data/                         ← 已构建好的回放栈（约 6 MB）
│   ├── slc.npy                   ← 2 × 500 × 500 complex64
│   ├── dem.npy                   ← 占位 DEM（全零）
│   ├── mask.npy                  ← 有效像元掩膜
│   ├── meta.json                 ← 工作台读取的元数据
│   ├── reference_los_mm.npy      ← 公开产品换算的 LOS，用作对照
│   └── replay_config.json        ← 构建参数与源文件校验和
├── scripts/
│   ├── build_case.py             ← 下载 / 校验 / 裁剪 / 构建回放栈
│   └── run_case.py               ← 导入、跑 D-InSAR、验证、出图
└── results/                      ← 已生成的结果
    ├── dingri_los_mm.npy
    ├── dingri_los_mm.tif         ← 可直接拖进 QGIS
    ├── summary.json
    └── figures/
```

`data/` 和 `results/` 已经提交进仓库，所以拿到仓库就能看结果。`source/` 目录不会提交，用来放 26 MB 的原始 GeoTIFF。

---

## 4. 路线 A：一条命令复现（推荐先做这个）

### 4.1 准备环境

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m insar_studio doctor
```

`doctor` 里 SNAPHU 显示不可用没关系，本案例使用内置解缠器。

### 4.2 重建回放栈

如果 `data/` 已经存在，可以跳过这一步。想亲手重建时：

```bash
python examples/01_dingri_2025_dinsar/scripts/build_case.py --download
```

脚本会：

1. 从 Zenodo 下载 `ASC_20250105_20250117.adf_quad.unw.geo.tif` 到 `examples/01_dingri_2025_dinsar/source/`；
2. 校验 MD5，必须是 `894d51cb912ff24ece4af8931d8ee57c`；
3. 裁剪 `row=569, col=1288, size=1500`；
4. 把 0 值无数据孔洞用最近邻填充；
5. 做 `sigma=10` 的高斯平滑；
6. 3 × 3 块平均降到 500 × 500；
7. 相位除以 10，写成两景复数 SLC；
8. 写出 `meta.json`、`replay_config.json` 和对照用的 `reference_los_mm.npy`。

下载慢或网络不稳时，可以先用浏览器下载，再指定本地文件：

```bash
python examples/01_dingri_2025_dinsar/scripts/build_case.py --source D:\data\ASC_20250105_20250117.adf_quad.unw.geo.tif
```

### 4.3 跑完整流程

```bash
python examples/01_dingri_2025_dinsar/scripts/run_case.py
```

脚本会创建一个临时工作区，导入 `data/`，按固定参数跑 `network → interferometry → inversion`，把结果乘以 10，和公开产品对比，写出 `results/`，最后删除临时工作区。想保留工作区排查问题：

```bash
python examples/01_dingri_2025_dinsar/scripts/run_case.py --keep
```

### 4.4 检查输出

```bash
python -c "import json;print(json.load(open('examples/01_dingri_2025_dinsar/results/summary.json',encoding='utf-8'))['validation_against_published_product'])"
```

应该看到相关系数约 1.0、RMSE 约 `1e-5 mm`。

---

## 5. 路线 B：在 Web 界面里手动复现

这一节是给“我想看着界面一步步点”的读者。脚本和界面走的是同一套引擎，结果应当一致。

### 5.1 启动

Windows 双击仓库根目录的 `start.bat`，或者：

```bash
python -m insar_studio serve --open
```

浏览器打开后，左侧是项目列表。

### 5.2 新建 D-InSAR 项目并导入

1. 点击左侧 **“+ 新建”**。
2. 项目名称填 `定日 2025 D-InSAR`。
3. 技术路线选 **D-InSAR**。
4. 来源选 **“导入本地数据”**。
5. 数据路径填：

   ```text
   D:\opencode_workspace\insar-studio\examples\01_dingri_2025_dinsar\data
   ```

   请换成你自己的仓库路径。
6. 勾选 **“导入的 SLC 已经配准”**。回放栈的两景本来就是同一网格，不需要再配准。
7. 创建项目，等待导入完成。

导入后进入 **“数据”** 页，核对：

- 景数：2；
- 尺寸：500 × 500；
- 日期：`2025-01-05`、`2025-01-17`；
- 地理参考：有，角点约 `86.893°E, 29.425°N`；
- 掩膜：边缘 2 像元被排除，其余有效。

如果日期或尺寸不对，不要继续跑，先检查 `data/meta.json`。

### 5.3 生成干涉对

进入 **“网络”** 页：

1. 主影像选 `2025-01-05`；
2. 从影像选 `2025-01-17`；
3. 生成网络。

D-InSAR 只会生成这一对，网络图里应该只有一条边。

### 5.4 设置干涉参数

进入 **“干涉”** 页，打开专家模式，按下面设置：

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `multilook` | 1 | 回放栈已经降到 500 × 500，再降会丢细节 |
| `filter` | `none` | 输入已经是平滑后的解缠相位对应的复数，不需要再滤波 |
| `coherence_window` | 5 | 小窗口保留形变边界 |
| `min_coherence` | 0.0 | 回放栈相干性接近 1，不设阈值 |
| `unwrap_coherence_min` | 0.0 | 避免把低相干但有效的远场裁掉 |
| `unwrap_method` | `quality` | 内置质量引导解缠 |
| `remove_ramp` | 关闭 | 同震形变本身就是长波信号，去趋势会吃掉真实形变 |
| `ramp_order` | 1 | 仅在开启去趋势时生效 |

然后运行干涉。

### 5.5 设置参考点并反演

进入 **“反演”** 页：

1. 空间参考选 **“手动指定”**；
2. 参考行 `row` 填 `24`；
3. 参考列 `col` 填 `63`；
4. 运行反演。

为什么是 `(24, 63)`？这是回放栈远场一个 LOS 接近 0 的稳定像元。D-InSAR 的位移是相对的，参考点选错会把一个常数偏移带进整张图。`(24, 63)` 在 `reference_los_mm.npy` 中约为 `-0.0005 mm`，适合做基准。

### 5.6 看成果

进入 **“成果”** 页：

- 位移图单位是 mm；
- 参考像元显示 `row 24 / col 63`；
- 图上的数值是**压缩后的**，需要乘以 10 才是公开 LOS；
- 可以在像元时序面板点选像元，核对峰值位置。

工作台不会自动乘 10，因为 `replay_phase_scale` 只是示例元数据，不是通用物理参数。导出后请自己乘，或者直接用 `results/dingri_los_mm.tif`。

### 5.7 导出

成果页支持导出 GeoTIFF 和数组。导出的 `disp_pair_mm.tif` 仍是压缩值，乘 10 后与 `results/dingri_los_mm.tif` 等价。

---

## 6. 参数为什么这样定

### 6.1 为什么相位要除以 10

源产品峰值约 1205 mm，对应约 87 个 2π 条纹（按 C 波段 5.55 cm 波长）。内置解缠器面对这么密的条纹，在噪声和降采样后会跳周。把相位整体除以 10，条纹密度降到约 9 个，解缠稳定；输出再乘 10 恢复。

这是一个**为了回放稳定而做的工程妥协**，不是物理处理步骤。它在 `meta.json` 里明确记录为 `replay_phase_scale: 10.0`，教程和结果里都要求乘回。

### 6.2 为什么要平滑和降采样

源产品 2532 × 3608，直接放进工作台会：

- 文件大、跑得慢；
- 保留大量高频噪声和解缠毛刺；
- 对教学没有额外价值。

`sigma=10` 高斯平滑 + 3 × 3 块平均把网格降到 500 × 500，像元约 250 m，同震形变的主要空间形态完整保留。代价是峰值被轻微削低、边界变平滑，所以本案例不用于科研量级的三维反演。

### 6.3 为什么不去趋势

同震形变的远场本来就有长波成分。`remove_ramp` 会拟合并减掉平面或二次面，等于把真实信号当轨道误差去掉。验证时相关系数会明显下降。正确做法是保留趋势，用稳定参考点定基准。

### 6.4 为什么参考点不是自动

自动参考点会选相干性高、时间稳定的像元，但回放栈的相干性几乎处处为 1，自动策略可能选到形变区边缘。手动 `(24, 63)` 是预先核对过的远场零值点，保证结果可复现。

---

## 7. 结果怎么看

### 7.1 图上能看到什么

- 震中附近一条近南北向的强正 LOS 带，峰值约 +1205 mm；
- 西侧一个负 LOS 瓣，约 −324 mm；
- 远场逐渐回到 0；
- 边缘 2 像元被掩膜，显示为空白。

这是典型的正断层同震形变在升轨 LOS 方向上的投影：断层两侧地表运动方向不同，投影到卫星视线方向后形成一正一负的瓣。**不要**把红色直接叫“抬升”、蓝色直接叫“沉降”；LOS 是三维位移在视线方向的投影，需要更多几何或先验才能分解。

### 7.2 哪些数字可以引用

可以引用：

- 形变空间形态；
- 峰值量级（约 1.2 m，受平滑影响略偏低）；
- 与公开产品的一致性（验证链路正确）。

不要引用：

- 把本案例峰值当作地震破裂的反演结果；
- 把 500 × 500、250 m 网格当作原始分辨率；
- 把回放栈当成原始 SLC 去讨论配准、去斜或 burst 拼接。

### 7.3 和公开产品的关系

`reference_los_mm.npy` 是从同一份公开解缠相位换算的 LOS，只做了裁剪、填充、平滑和降采样。工作台输出与它的差异只来自：

1. 解缠器的数值路径；
2. 参考点的常数基准；
3. 相位压缩与恢复的浮点误差。

因此残差在 `1e-5 mm` 量级是预期结果。它说明链路自洽，不说明独立观测精度。

---

## 8. 如果要从原始 SLC 开始

本案例刻意没有下载原始 SLC。原因：

- 单景 Sentinel-1 SLC 数 GB，两景加轨道文件对教学仓库太重；
- 工作台不做 TOPS 配准、去斜（deramp）和 burst 拼接；
- 直接拿未配准的 TOPS SLC 进来，结果会是噪声，不是形变。

真实原始数据路径：

1. 在 Copernicus Data Space Ecosystem 或 ASF Vertex 注册账号；
2. 下载 2025-01-05 和 2025-01-17 的升轨 SLC，以及精密轨道 `.EOF`；
3. 用 SNAP（TOPS Coregistration → Interferogram → TopoPhaseRemoval → Goldstein → Unwrap → Terrain Correction）或 ISCE2 / Gamma 处理；
4. 导出配准后的复数栈或解缠相位；
5. 再导入 InSAR Studio 做参考点、位移转换、出图和验证。

这条路径需要账号、磁盘和时间，教程不把它伪装成“一键完成”。本案例的价值是把“拿到产品之后怎么用”讲清楚。

---

## 9. 常见问题

**Q：为什么 `data/dem.npy` 全是 0？**

回放栈的相位已经是解缠后的差分相位，不再需要 DEM 去地形。全零 DEM 只是让导入器有合法输入，不参与物理计算。`meta.json` 里 `already_coregistered: true`，工作台不会再用 DEM 做配准。

**Q：为什么工作台图上的峰值只有 120 左右？**

你看的是压缩后的值。乘 10 才是毫米。`results/dingri_los_mm.tif` 已经乘好。

**Q：为什么参考点不是自动？**

自动策略在常数相干性场上没有区分度。手动 `(24, 63)` 是预先验证的远场零点。

**Q：为什么 `remove_ramp` 关闭后图还有大范围颜色？**

那是真实同震长波，不是轨道 ramp。判断 ramp 要看它是否与形变无关、是否跨整个场景线性/二次分布；不要看到大范围颜色就习惯性去趋势。

**Q：跑出来和 `results/` 不完全一样？**

检查 Python / NumPy / SciPy 版本，以及是否用了相同的 `multilook`、`filter`、`unwrap_method` 和参考点。解缠器在不同版本间可能有微小差异。

**Q：可以把这个案例写进论文吗？**

可以引用数据源和本仓库作为教学复现；不要把派生回放栈当作独立观测。科研使用请引用原始论文和 Zenodo 产品，并遵守 CC-BY-4.0 署名要求。

---

## 10. 引用与许可

数据：

> Geodetic Observations and Seismogenic Structures of the 2025 Mw 7.0 Dingri Earthquake: The Largest Normal Faulting Event in the Southern Tibet Rift. Zenodo. https://doi.org/10.5281/zenodo.14836721. License: CC-BY-4.0.

本案例代码和文档随 InSAR Studio 以 MIT 许可发布；`data/` 中的派生数组沿用源数据的 CC-BY-4.0，分发时请保留上述署名和 DOI。

变换细节、文件校验和和逐步命令见 [PROVENANCE.md](PROVENANCE.md)。
