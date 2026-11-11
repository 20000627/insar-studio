# InSAR Studio 1.2.0 分享包说明

这份说明面向收到发布包的人。建议先读完本页，再按里面的步骤解压、校验和启动。

## 这是什么

InSAR Studio 是一个本地运行的时序 InSAR 形变处理工作台。它把「数据接入 → 配准 → 干涉网络 → 差分干涉 → 大气校正 → 时序反演 → 成果输出」串成一条可复现的处理链，并提供中文分步向导。

支持四条路线：

| 路线 | 适用场景 | 最少影像 |
| --- | --- | ---: |
| SBAS-InSAR | 面状沉降、滑坡、城市形变 | 8 |
| PS-InSAR | 城市建筑、岩石裸露区的点目标监测 | 8 |
| DS-InSAR | 植被、裸土等低相干区 | 10 |
| D-InSAR | 同震、滑坡等单次事件 | 2 |

包内自带合成演示数据生成器，依赖装好后不联网也能跑完四条路线。真实数据需要先在外部工具完成 SAR 预处理，再通过界面或命令行导入。

## 包内有什么

| 路径 | 内容 |
| --- | --- |
| `insar_studio/` | 程序源码、页面模板、前端静态资源 |
| `docs/` | 新手指南、用户手册、使用指南、技术说明和界面截图 |
| `examples/` | 两个真实案例：定日地震 D-InSAR、白格滑坡 SBAS，含数据、结果和教程 |
| `tests/` | pytest 回归测试 |
| `scripts/bootstrap.py` | 首次启动时的依赖检查和虚拟环境创建 |
| `scripts/package_release.py` | 重新生成发布包和校验文件 |
| `start.bat` / `start.sh` | Windows / Linux、macOS 一键启动 |
| `README.md` | 项目概览、快速开始和命令摘要 |
| `RELEASE.md` | 本说明 |
| `LICENSE` | MIT 许可证 |
| `pyproject.toml`、`requirements.txt` | 包元数据和依赖清单 |
| `.env.example`、`.gitignore` | 环境变量示例和版本控制忽略规则 |

发布包不包含 `workspace/`、测试工作区、`.venv/`、缓存、日志、凭据和任何真实原始影像。第一次启动会在本地新建 `workspace/`，演示项目产生的数据都留在本机。

包内的两个真实案例是公开产品派生的回放栈，不是原始 SLC 重处理。`examples/*/data/` 和 `results/` 是刻意保留的示例数据，便于离线复现；`source/`、`workspace/` 和原始大文件不会进入发布包。

## 快速开始

### 1. 校验下载的文件

发布包旁边应有一个同名 `.sha256` 文件，例如：

```text
insar-studio-1.2.0.zip
insar-studio-1.2.0.zip.sha256
```

Windows PowerShell：

```powershell
Get-FileHash .\insar-studio-1.2.0.zip -Algorithm SHA256
Get-Content .\insar-studio-1.2.0.zip.sha256
```

Linux / macOS：

```bash
sha256sum -c insar-studio-1.2.0.zip.sha256
```

把计算的哈希与 `.sha256` 文件中的第一段比较，忽略大小写。两者一致才继续；不一致说明文件在下载或传输中损坏，请重新获取，不要直接运行。

### 2. 解压

解压到本机一个路径较短、没有特殊字符的目录，例如 `D:\insar-studio`。不要在压缩软件的预览窗口里直接运行，也不要放在需要管理员权限的系统目录。

### 3. 启动

Windows：双击根目录的 `start.bat`。第一次使用建议先运行：

```powershell
start.bat --check
```

它只检查 Python、64 位环境、依赖和工作区写入权限，不启动服务。如果只提示缺少依赖，继续双击 `start.bat`，让脚本自动创建 `.venv` 并安装；如果提示 Python 版本过低、高于 3.11 或不是 64 位，先安装 64 位 Python 3.11 再重试。

Linux / macOS：

```bash
bash start.sh --check
bash start.sh
```

首次启动时，`scripts/bootstrap.py` 会在项目目录创建 `.venv` 并安装 `requirements.txt`，需要联网，通常只需一次。之后可以离线运行。终端会打印实际访问地址，例如 `http://127.0.0.1:8765`，并尝试自动打开浏览器。保持命令行窗口打开；关闭窗口或按 `Ctrl+C` 停止服务。

### 4. 跑通一个演示

打开页面后，在左侧「项目库」选择任一演示项目，进入「成果」步骤，点击「一键跑完全流程」。也可以在终端运行：

```powershell
python -m insar_studio demo --name "分享包验收" --technique sbas
```

如果启动器已经创建了 `.venv`，用虚拟环境里的解释器执行：

```powershell
.\.venv\Scripts\python.exe -m insar_studio demo --name "分享包验收" --technique sbas
```

Linux / macOS 对应 `.venv/bin/python`。命令会打印项目 ID、耗时、速率中位数和真值 RMSE。

### 5. 跑两个真实案例

演示跑通后，直接运行发布包里的真实案例：

```powershell
python examples/01_dingri_2025_dinsar/scripts/run_case.py
python examples/02_baige_landslide_sbas/scripts/run_case.py
```

第一个案例用 2 景回放数据跑 D-InSAR，几分钟内得到同震 LOS 形变图和与公开产品的对照；第二个案例用 41 历元跑 SBAS，得到速率图和峰值时间序列。两个案例的 `data/` 和 `results/` 已包含在包内。白格案例重建数据需要额外安装 `h5py`，只跑结果不需要。

零基础操作路径见 `docs/新手指南.md`，案例细节见 `examples/README.md`。

## 验收清单

在解压目录依次执行：

```powershell
start.bat --check
python -m insar_studio doctor
python -m insar_studio demo --name "验收 SBAS" --technique sbas
python examples/01_dingri_2025_dinsar/scripts/run_case.py
python examples/02_baige_landslide_sbas/scripts/run_case.py
python -m pytest -q
```

预期结果：

- `--check` 返回成功，并列出依赖和 Python 版本；首次运行如果只提示缺少依赖（退出码 3），执行一次 `start.bat` 完成安装；
- `doctor` 不报告必需依赖缺失，SNAPHU 显示为可选；
- 演示命令正常结束，生成 `products/` 和报告；
- 定日案例输出相关系数接近 1、去常数偏移后的 RMSE 在 `1e-5 mm` 量级；
- 白格案例输出 RMSE 约 `0.20 mm`、相关系数约 `0.99999`、末历元峰值约 `74.1 mm`；
- 全量测试为 **219 passed**，通常 40–80 秒。测试是可选验收项，需要先安装 pytest：`python -m pip install pytest`，或 `python -m pip install -e ".[dev]"`。如需包含示例完整性检查，请安装 `h5py`（仅重建白格数据时需要，运行已提交数据不需要）。

启动服务后，浏览器能打开页面、切换七步、查看成果图件，即表示 Web 界面可用。

## 环境要求

| 项目 | 最低 | 推荐 |
| --- | --- | --- |
| 操作系统 | Windows 10/11、Linux、macOS | Windows 10/11 或 Ubuntu 22.04 |
| Python | 3.9，64 位 | 3.11，64 位 |
| 内存 | 4 GB 可用 | 8 GB 以上；真实大栈 16 GB 以上 |
| 磁盘 | 2 GB 程序空间 | 20 GB 以上，另加数据空间 |
| 浏览器 | 现代 Chrome、Edge、Firefox | Chrome 或 Edge |

高于 3.11 的 Python 可以尝试，但部分依赖可能没有对应轮子；启动器会提示而不会直接阻断。没有安装 SNAPHU 不影响默认流程。

## 已知限制

- 合成演示的指标只用于软件回归，不能当作真实数据的精度承诺。
- PS 和 DS 是面向算法验证的简化实现，不包含 StaMPS、MintPy 级别的完整逐点相位模型；真实业务应与成熟实现交叉验证。
- 真实 Sentinel-1 处理仍依赖外部 TOPS 配准；本工具可用本地 EOF 精密轨道估计垂直基线，并在几何元数据齐全时做距离-多普勒地理编码，但不替代 SNAP、ISCE、Gamma 的预处理链。
- 包内两个真实案例是公开产品派生的回放栈：定日案例做了裁剪、平滑、降采样和相位压缩，白格案例的时间轴是重建的。它们用于教学和链路验证，不是原始 SLC 处理，也不构成新的地质结论。
- 不包含大气数值天气预报自动下载、GNSS 融合、角反射器识别和断层反演。GACOS 采用官方申请后邮件导入的方式。
- 没有用户认证和多用户隔离，服务默认只监听本机，请不要直接暴露到公网。
- 所有数据保存在本地 `workspace/`，删除项目会删除对应目录，请自行备份。

## 文档入口

| 文档 | 适合什么时候看 |
| --- | --- |
| `README.md` | 先了解功能、快速开始和常用命令 |
| `docs/新手指南.md` | 零基础：下载、校验、解压、启动、导入、跑案例、备份和常见问题 |
| `docs/用户手册.md` | 第一次使用，按最短路径跑通 |
| `docs/使用指南.md` | 查影像类型、处理原理、参数表和常见问题 |
| `docs/技术说明.md` | 审查架构、算法、API、安全边界和复现方法 |
| `examples/README.md` | 两个真实案例的入口、数据和限制 |
| `LICENSE` | MIT 许可证全文 |

## 许可

代码和文档使用 MIT 许可证，允许复制、修改、分发和商用，前提是保留版权和许可声明。演示数据由代码生成，可以随代码发布。

`examples/01_dingri_2025_dinsar/data/` 和 `examples/02_baige_landslide_sbas/data/` 是 CC-BY-4.0 公开产品的派生数组，不适用 MIT：分发时请保留各案例 `PROVENANCE.md` 中的 DOI、标题、作者和许可说明，并明确标注“派生回放栈”。使用 Sentinel-1、GACOS、商业 DEM 或其他第三方数据时，请遵守各自许可，不要分发没有授权的内容。
