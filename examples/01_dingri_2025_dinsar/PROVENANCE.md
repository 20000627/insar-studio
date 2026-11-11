# 数据来源与变换记录：定日 2025 D-InSAR 案例

本文件记录 `data/` 中每个数组的来源、变换步骤、校验和，以及 `results/` 的生成方式。目标是让另一个人不看聊天记录，也能判断这些文件能不能用、怎么重建。

## 1. 源数据

| 项目 | 值 |
| --- | --- |
| 仓库 | Zenodo |
| 记录 | 14836721 |
| DOI | `10.5281/zenodo.14836721` |
| 标题 | Geodetic Observations and Seismogenic Structures of the 2025 Mw 7.0 Dingri Earthquake: The Largest Normal Faulting Event in the Southern Tibet Rift |
| 文件 | `ASC_20250105_20250117.adf_quad.unw.geo.tif` |
| 大小 | 27,545,235 字节（26.27 MiB） |
| MD5 | `894d51cb912ff24ece4af8931d8ee57c` |
| 许可 | CC-BY-4.0 |
| 下载 URL | `https://zenodo.org/api/records/14836721/files/ASC_20250105_20250117.adf_quad.unw.geo.tif/content` |

源文件是 GAMMA `data2geotiff` 输出的单波段 float32 解缠相位，地理参考为 WGS 84：

- 尺寸 2532 × 3608；
- 像元 0.00083333334°；
- 左上角约 `85.8198611°E, 29.8993055°N`；
- 数值范围约 −75.31 到 297.16 rad；
- 约 36.3% 的像元为 0（无数据/掩膜）。

`scripts/build_case.py` 会先校验 MD5，再读取这些标签。MD5 不匹配会直接退出，避免用被截断的文件构建。

## 2. 构建步骤

脚本：`scripts/build_case.py`

1. 读取 GeoTIFF 单页和 `ModelPixelScaleTag` / `ModelTiepointTag`。
2. 裁剪 `row=569, col=1288, size=1500`，得到 1500 × 1500 窗口，覆盖主形变瓣。
3. 有效像元定义为 `isfinite & != 0`；裁剪窗口内有效比例约 99.4%。
4. 用 `scipy.ndimage.distance_transform_edt` 做最近邻填充，填掉 0 值孔洞。
5. `scipy.ndimage.gaussian_filter(sigma=10.0)` 平滑。
6. 3 × 3 块平均，得到 500 × 500。
7. 相位乘以 `0.1`（`PHASE_SCALE`），写入 `slc.npy` 的第二景；第一景相位为 0。
8. 振幅用确定性的余弦纹理构造，范围约 0.55–1.25，不含随机数。
9. 掩膜去掉最外 2 像元。
10. 用 `wavelength=0.055465763 m` 把平滑后的相位换算成 `reference_los_mm.npy`。
11. 写出 `meta.json` 和 `replay_config.json`。

地理网格：

```text
corner_lon = 85.8198611 + 1288 × 0.00083333334 = 86.89319444192
corner_lat = 29.8993055 − 569 × 0.00083333334 = 29.42513882954
lon_step   = 0.00083333334 × 3 = 0.0025
lat_step   = −0.0025
```

## 3. 回放栈的物理含义

`slc.npy` 的两景满足：

```text
phase(slc[1]) − phase(slc[0]) = 0.1 × smoothed_unwrapped_phase
```

因此工作台解缠、去参考点后得到的位移，是公开 LOS 的 `1/10`。`meta.json` 中：

```json
"replay_phase_scale": 10.0
```

`run_case.py` 在读取 `products/arrays/disp_pair_mm.npy` 后乘 10。`results/dingri_los_mm.npy` 和 `.tif` 已经是乘 10 的值。

这不是原始 SLC。它没有：

- 原始振幅；
- 原始相位噪声；
- 配准误差；
- TOPS burst 不连续性；
- 轨道和 DEM 误差。

它适合验证“相位 → 解缠 → 参考点 → LOS 位移 → 出图”这条链路，不适合评估原始 SLC 处理质量。

## 4. 文件校验和

`data/`：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `slc.npy` | 4,000,128 | `fae16f6649464d2ab472495b946f1e253842a07309045f081101a2b5cc68c63d` |
| `dem.npy` | 1,000,128 | `403734f34d35d7ef05b799179206eb08d9b06c009e8f239a38e978a437a6f734` |
| `mask.npy` | 250,128 | `68c7811118a88f53df85c970037f315509ce789630aecfbe82dcd1bb184ebd11` |
| `meta.json` | 1,765 | `932bd9c035708ee120f975cf3c6713deeab692996fd31e12df0fcd987b63e048` |
| `reference_los_mm.npy` | 1,000,128 | `33fa045448144196f5be65fcdec4869071849a03eea98cfdcea53ff506dcd513` |
| `replay_config.json` | 1,909 | `790f95ca30344f501392f057f7c4e746e781a117b342657bfcd03859a4e429b1` |

`results/`：

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| `dingri_los_mm.npy` | 1,000,128 | `d6ec6322b6b8b7cf22d0e787529c449e153dfb513c40359505825525795a90eb` |
| `dingri_los_mm.tif` | 1,250,467 | `3816f2f5ab35c9bf4d3c4ee351ea70c3a6c96291729c13176941d8107bec45fd` |
| `summary.json` | 1,671 | `73778b4897ec718f5ffbc2b10dd54e5d3536800c8bf1dadaf18d81afdd5f32d3` |
| `figures/dingri_los_mm.png` | 136,745 | `f930ede2ee303c26cd045f8f311a8249ab810157e7edf6a4fa22e14c667ad512` |
| `figures/dingri_validation.png` | 704,287 | `65630cfab1e0c32cd77ca03272dc6c66d472fced634570bc94d23e78ce924465` |

`summary.json` 会在重新运行 `run_case.py` 后带有新的 `elapsed_s`，所以它的哈希会变；数组和图件在相同依赖版本下应保持一致。

## 5. 处理参数

| 参数 | 值 |
| --- | --- |
| 技术路线 | `dinsar` |
| 干涉对 | 0–1（2025-01-05 → 2025-01-17） |
| `multilook` | 1 |
| `filter` | `none` |
| `coherence_window` | 5 |
| `min_coherence` | 0.0 |
| `unwrap_coherence_min` | 0.0 |
| `unwrap_method` | `quality` |
| `remove_ramp` | `false` |
| `reference_mode` | `manual` |
| 参考像元 | `(24, 63)` |

## 6. 验证

`run_case.py` 把工作台输出乘 10 后与 `reference_los_mm.npy` 逐像元比较：

| 指标 | 值 |
| --- | --- |
| 有效像元比例 | 98.4% |
| 中位常数偏移 | 约 `0.00049 mm` |
| 去偏移后 RMSE | 约 `1.13e-05 mm` |
| 去偏移后 95% 绝对残差 | 约 `2.00e-05 mm` |
| 相关系数 | `0.9999999999999952` |

残差量级说明工作台的相位到位移转换、参考点和缩放比例都正确。它不是独立精度评估。

## 7. 许可与署名

源数据 CC-BY-4.0。分发 `data/`、`results/` 或由其改写的产品时，请保留：

- DOI `10.5281/zenodo.14836721`；
- 数据集标题；
- CC-BY-4.0 许可说明；
- 本文件的变换说明，明确标注“派生回放栈”。

仓库代码和文档以 MIT 许可发布；两者许可范围不同，不要混用。
