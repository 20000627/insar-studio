# 真实案例

这里放两个可以直接跑通的真实数据案例。它们不是合成演示，而是从公开的 CC-BY-4.0 产品派生的**回放栈**：相位或位移来自真实观测产品，但为了教学体积和稳定性做了裁剪、平滑、降采样或时间轴重建。每个案例的 `PROVENANCE.md` 都写清楚了哪些是原始观测、哪些是派生变换。

| 案例 | 技术路线 | 数据源 | 规模 | 你会得到 |
| --- | --- | --- | --- | --- |
| [01 定日 2025 地震 D-InSAR](01_dingri_2025_dinsar/) | D-InSAR | Zenodo 14836721，升轨解缠相位，CC-BY-4.0 | 2 景，500 × 500，约 6 MB | 同震 LOS 形变图、GeoTIFF、与公开产品的逐像元验证 |
| [02 白格滑坡 SBAS](02_baige_landslide_sbas/) | SBAS | Zenodo 10605366，位移时间序列，CC-BY-4.0 | 41 历元，51 × 95，约 2.4 MB | LOS 速率图、峰值像元时间序列、与公开产品的对照 |

## 怎么选

- 第一次用，先做 **01 定日**。它只有两景、参数少，几分钟就能看到一张同震形变图。
- 想做时序、看速率和时间序列，做 **02 白格**。它会走完整的干涉网络和 SBAS 反演。
- 想学界面操作，两个案例的 README 都给了 Web 界面逐步点击路径。
- 想批量复现，两个案例都有 `scripts/run_case.py`，一条命令跑完。

## 快速运行

在仓库根目录：

```bash
python -m pip install -r requirements.txt
python examples/01_dingri_2025_dinsar/scripts/run_case.py
python examples/02_baige_landslide_sbas/scripts/run_case.py
```

`data/` 和 `results/` 已经提交，所以不重建数据也能直接跑。想从头下载和构建：

```bash
python examples/01_dingri_2025_dinsar/scripts/build_case.py --download
python examples/02_baige_landslide_sbas/scripts/build_case.py --download
```

第二个案例的 `Dt.mat` 是 MATLAB v7.3，重建数据需要额外安装 `h5py`：

```bash
python -m pip install h5py
```

## 重要限制

两个案例都是**派生回放栈**，不是原始 SLC 重处理：

- 工作台不做 TOPS 配准、去斜和 burst 拼接；
- 定日案例的相位来自公开解缠产品，并做了平滑、降采样和相位压缩；
- 白格案例的位移来自公开时间序列，时间轴是重建的；
- 案例验证的是“工作台链路是否忠实”，不是“工作台能否替代 SNAP / ISCE / Gamma”。

真实科研或业务使用，请从原始 SLC 开始，用成熟生产链处理，再用本工作台做参考点、出图和交叉验证。

## 数据许可

两个数据源都是 CC-BY-4.0。分发 `data/`、`results/` 或改写产品时，请保留各案例 `PROVENANCE.md` 中的 DOI、标题、作者和许可说明。仓库代码和文档是 MIT 许可，两者不要混用。
