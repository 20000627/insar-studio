# 数据来源与变换记录：白格滑坡 SBAS 案例

本文件记录 `data/` 中每个数组的来源、变换步骤、校验和，以及 `results/` 的生成方式。目标是让另一个人不看聊天记录，也能判断这些文件能不能用、怎么重建。

## 1. 源数据

| 项目 | 值 |
| --- | --- |
| 仓库 | Zenodo |
| 记录 | 10605366 |
| DOI | `10.5281/zenodo.10605366` |
| 标题 | Displacement time series of the Baige landslide |
| 作者 | Zhiqiang Xiong |
| 文件 | `disp_data.rar` |
| 大小 | 5,617,416 字节（5.36 MiB） |
| MD5 | `0540704e40a1924c4a726aea6482855b` |
| 许可 | CC-BY-4.0 |
| 下载 URL | `https://zenodo.org/api/records/10605366/files/disp_data.rar/content` |

RAR 内包含：

```text
disp_data/disp_par2     GAMMA DIFF&GEO DEM/MAP 参数文件
disp_data/Dt.mat        MATLAB v7.3，Dt = (199, 95, 51) float64
disp_data/mask.mat      MATLAB v5，mask = (51, 95) uint8，取值 0/1
```

`disp_par2` 给出的地理编码：

```text
width:                95
nlines:               51
corner_lat:     31.08736
corner_lon:     98.69603
post_lat:   -1.8518519e-04
post_lon:    1.8518519e-04
ellipsoid: WGS 84
```

数据包没有说明：

- 采集日期；
- 卫星平台和轨道方向；
- 位移符号约定（本案例按 LOS 正负解释）；
- 单位（本案例按 mm 解释）。

这些缺失不影响“用同一份位移构造回放栈再反演回来”的自洽验证，但限制了绝对速率的解释。

## 2. 构建步骤

脚本：`scripts/build_case.py`

1. 下载 RAR，校验 MD5。
2. 用系统 `tar` 解压，失败时尝试 `rarfile`。
3. 用 `scipy.io.loadmat` 读取 `mask.mat`；`Dt.mat` 是 v7.3，回退到 `h5py`。
4. 读取 `Dt` 为 `(199, 95, 51)`，转置空间轴为 `(199, 51, 95)`，与 `mask` 的 `(51, 95)` 对齐。
5. 取索引 `0, 5, 10, …, 195, 198`，共 41 个历元。
6. 按 `2018-12-14 + 6 × 源索引` 生成重建日期。
7. 用 `wavelength=0.055465763 m` 把位移（按 mm）转成相位：`phase = 4π/λ × disp / 1000`。
8. 用确定性余弦纹理构造振幅，范围约 0.55–1.25，不含随机数。
9. 写出 41 景 complex64 SLC。
10. 写出 `meta.json`、`replay_config.json`、`truth_disp_mm.npy`。

原始 `Dt` 的数值范围是 0–173.54 mm；173.54 mm 的瞬态峰值出现在掩膜外，掩膜内最大值是 75.51 mm，出现在末历元。回放栈保留完整 `Dt` 的所选历元，掩膜外的值不参与反演。

## 3. 重建时间轴的准确含义

`meta.json` 中：

```json
"dates": ["2018-12-14", "2018-12-20", ...],
"dates_source": "reconstructed",
"sampling_days": [0, 30, 60, ...],
"reconstructed_time_axis": true,
"reconstructed_start": "2018-12-14",
"reconstructed_interval_days": 6,
"source_epoch_indices": [0, 5, 10, ..., 195, 198]
```

41 个历元的源索引间隔是 5，对应重建轴上的 30 天；最后一个间隔是 3 个源索引，对应 18 天。SBAS 网络使用这些采样天数计算时间基线，`max_temporal_baseline=60` 允许每个历元连接前后各两个。

这条轴是**假设**，不是从数据包恢复的真实采集日期。它只用于：

- 构建设计矩阵；
- 把位移除以时间得到速率；
- 让季节项有频率含义。

它不用于：

- 断言某次滑坡加速发生在某个真实日期；
- 把 8.24 mm/yr 当作白格滑坡的绝对速率；
- 与真实 SAR 采集序列做逐景对齐。

## 4. 回放栈的物理含义

`slc.npy` 的第 `i` 景满足：

```text
phase(slc[i]) = 4π/λ × truth_disp_mm[i] / 1000
```

工作台从相邻对相位差中恢复的位移，在参考点归一化后应等于 `truth_disp_mm` 的相对值。因为相位生成和反演使用同一波长，波长选择不会改变位移结果；它只影响条纹密度。

这不是原始 SLC。它没有：

- 原始振幅和相位噪声；
- 配准误差；
- 轨道和 DEM 误差；
- 大气延迟；
- 时间去相干。

它适合验证“相位栈 → 网络 → 解缠 → SBAS → 时间序列 → 速率”的链路，不适合评估原始 SLC 处理质量。

## 5. 文件校验和

`data/`：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `slc.npy` | 1,589,288 | `48e0acee59083ac3aa263297507ca4c8e5a26aa5fba0d8304ac7c2da67033f50` |
| `dem.npy` | 19,508 | `91d4f0ff46469fadf3a8dbd4464337feb4879f95854c9a7fddc0b0068058283e` |
| `mask.npy` | 4,973 | `f34de3b63f5e17b7d22372b3c755c515fc91284532f694ef1a640834e6a280cf` |
| `meta.json` | 4,776 | `6ec9ceaff4096c8a59661fdb3a759f4a1745268d8b706de6fdd0425e97d1616c` |
| `truth_disp_mm.npy` | 794,708 | `53d9bee17a1a5dc521a34e5de5ebe666fefbe0cd928cab76602f3e60465e3726` |
| `replay_config.json` | 2,453 | `4719c4a1dc5bcd4a6391025efae1831d42b53acf0f60c11396c27604fac00301` |

`results/`：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `baige_disp_ts_mm.npy` | 794,708 | `92a7a811334cc974ddf401fa6dfd62531e1dc3efbef36d8fcb7200438487969b` |
| `baige_velocity_mm_yr.npy` | 19,508 | `4cf577c7b4c3b19c0ae438b9a40e15ff7bfe099d82163a644b10f378a62aef2f` |
| `baige_velocity_mm_yr.tif` | 24,346 | `28704923f783ed6b90ba63fb83bf5bd2c49b9a73700eb166af007db677ee0ca1` |
| `baige_disp_last_mm.tif` | 24,334 | `1e74c97b5dfe5f05c320ed8a8b63eebc222503b42a857b062461207a94623e92` |
| `summary.json` | 2,354 | `71930b4ee627e120915cf3f61d1c112cbcdf86074b85c126d8c59c6509ab8f87` |
| `figures/baige_velocity.png` | 52,656 | `cc12e2eb79ab26ba70f6407733a8902de71c00d107d015d71b6f6d7dd36cd6bf` |
| `figures/baige_timeseries.png` | 96,590 | `29b8594675f54fe7d6310dd70955e37e88477ddb8332d28b9cfc5c69f8cceff8` |

`summary.json` 会在重新运行 `run_case.py` 后带有新的 `elapsed_s`，哈希会变；数组和图件在相同依赖版本下应保持一致。

## 6. 处理参数

| 参数 | 值 |
| --- | --- |
| 技术路线 | `sbas` |
| 历元数 | 41 |
| 网络 | 最大时间基线 60 天，最大 300 对，实际 79 对 |
| `multilook` | 1 |
| `filter` | `none` |
| `coherence_window` | 5 |
| `min_coherence` | 0.0 |
| `unwrap_coherence_min` | 0.0 |
| `unwrap_method` | `quality` |
| `remove_ramp` | `false` |
| `estimate_dem_error` | `false` |
| `aps_method` | `off` |
| `aps_iterations` | 0 |
| `regularization` | 0.0 |
| `robust_iterations` | 1 |
| `reference_mode` | `manual` |
| 参考像元 | `(9, 30)` |

## 7. 验证

`run_case.py` 把工作台位移时间序列和 `truth_disp_mm.npy` 都减去参考像元 `(9, 30)` 的序列，然后在 `valid & mask` 的全部样本上比较：

| 指标 | 值 |
| --- | --- |
| 有效样本 | 73,800（41 × 1800） |
| RMSE | 0.2036 mm |
| 95% 绝对残差 | 0.3315 mm |
| 相关系数 | 0.99998755 |
| 末历元峰值（工作台） | 74.0851 mm |
| 末历元峰值（公开） | 75.5097 mm |
| 峰值差 | −1.4247 mm |

残差量级说明 SBAS 链路、参考点和参数设置正确。它不是独立精度评估。

## 8. 许可与署名

源数据 CC-BY-4.0。分发 `data/`、`results/` 或由其改写的产品时，请保留：

- DOI `10.5281/zenodo.10605366`；
- 数据集标题和作者 Zhiqiang Xiong；
- CC-BY-4.0 许可说明；
- “重建时间轴”的说明，明确绝对速率不是新的地质结论。

建议同时引用背景论文：

> Xiong, Z., Feng, G., Feng, Z., Miao, L., Wang, Y., Yang, D., & Luo, S. (2020). Pre- and post-failure spatial-temporal deformation pattern of the Baige landslide retrieved from multiple radar and optical satellite images. Engineering Geology, 279, 105880. https://doi.org/10.1016/j.enggeo.2020.105880

仓库代码和文档以 MIT 许可发布；两者许可范围不同，不要混用。
