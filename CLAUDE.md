# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

在 **Windows、macOS 或 Linux（含 IQ9075 边缘设备）** 上驱动 AgileX Piper 机械臂，并把它接到集群上的 OpenPI 风格策略服务器：读取臂状态 + 两路 Intel RealSense D405 彩色图像（第三视角 + 腕部视角）→ 通过 SSH 隧道经 WebSocket(msgpack) 发送 observation → 收到一个 action chunk（已验证为 `(50, 12)`）→ 整段执行 → 再观测、再推理，直到 `Ctrl+C`。

权威文档：`DEVELOPER_GUIDE.md`（协议、契约、设备问题）和 `USER_GUIDE.md`（操作步骤）。中文版 `开发文档.md` / `用户文档.md` 是同内容的翻译——**改动文档时要同时更新中英文两份**。

没有测试套件、没有构建步骤、没有依赖清单文件。运行时依赖：`numpy`、`msgpack`、`websockets>=13`、`opencv-python`（`cv2`）、`pyrealsense2`（Linux/Windows；macOS 改用 `pyobjc-framework-AVFoundation` + `pyusb`）、`python-can`、官方 `pyAgxArm`（`pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"`），录制/导出还要 `pandas`、`pyarrow`。两台 D405 按序列号识别（`DEFAULTS` 里的 `third_view_serial` / `wrist_serial`，可在 `team_config.json` 覆盖），不按 OpenCV 索引，重启后顺序不会变。**深度不能用来认相机**：同型号两台挂在一个集线器上会互相干扰深度，曾经因此把两路标反；角色只来自配置，用 `_realsense_test.py` 的带标签截图或 `cameras --motion-check` 验证。

本机可以是 Windows 或 macOS，直连机械臂，经 SSH 隧道连集群上的策略服务器。采集用 `collect.py`。Mac 上命令用 `.venv/bin/python`。

## 常用命令（在本目录下运行；Mac 上把 `python` 换成 `.venv/bin/python`）

```powershell
python team_config.py show      # 列出还没填的键（真实值在 team_config.json，它不入库）
python team_config.py arm       # 只读臂状态，不动臂（检查是否在线）
python team_config.py run       # 提交/复用 slurm 任务 serve-policy，等 "listening on"，开 ssh -L，启动连续控制
python team_config.py once      # 服务器和转发仍在时再跑一轮连续控制
python team_config.py run --observe-only --cycles 3  # 只验证链路：发观测、打印回来的 chunk，不发运动帧
python team_config.py down      # 取消 serve-policy 任务并关闭本地 ssh -N
python team_config.py job|serve # 仅打印手动启动服务器的命令
python team_config.py setup-remote  # server_mode=simple：把测试服务器拷到 remote_code_dir 并建 venv
python simple_policy_server.py --port 8000 [--mode demo|nudge|replay --episode F]  # 脚本化测试策略，默认关节 demo
python episode_recorder.py --task "..."  # 先自动移到 READY_DEG，再拖动示教录制（相机+反馈，实时预览窗口），s/f/d 结束；--no-ready/--no-preview
python demo.py [--task "..."] [--out DIR]  # 演示窗口：两路相机 + 从臂状态；r 主从录制（s/f/d 结束，存 recording.mp4；--out 默认 episodes），p 确认主臂断电后只在从臂回放最新成功录制（存 replay_*.mp4），q 退出
python collect.py --leader --task "..."  # 主从遥操作录制：从臂跟随同一总线上的主臂，不动臂，另存 lead_* 列
python policy_link.py replay [episodes/episode_...] [--loop]  # --loop 一直重复直到 Ctrl+C； 本机回放一条（默认最新），前后都到 READY_DEG，不需要服务器，相机只做预览窗口（会动臂）
python episode_check.py episodes/episode_...   # 时间线和曲线检查
python dataset_export.py episodes/episode_...  # 成功的 episode 导出 0.05 s 网格的 12 维动作和 mask；--self-check 不连臂
python prepare_dataset.py          # 把所有成功 episode 整理成 datasets/piper_finetune；--self-check 不连臂
python collect.py --task "..."     # 录一条，s 之后检查、导出并重建微调数据集
python team_config.py run --server replay|openpi [--ready]  # 覆盖 server_mode；--ready 先展开到 READY_DEG
python policy_link.py ready     # 只把臂展开到准备姿态（会动臂）
python policy_link.py home      # 收回到休息位稍上方 home_deg（默认 J2 10°、J3 −5°；会动臂）
python policy_link.py cameras   # 按序列号列出两台 D405，标出第三视角 / 腕部视角（角色只来自 team_config.json）
python policy_link.py cameras --motion-check  # 各自拍一张，动一下臂再回车，跟着动的那台就是腕部视角
python policy_link.py preview   # 实时预览，左上角是 224×224 模型输入（两路并排），q 退出
python policy_link.py serve     # 本地占位服务器 127.0.0.1:8765，只检查 observation 形状、返回占位动作
python policy_link.py once [--port N] [--freq HZ] [--speed PCT]  # 关节动作：目标每秒更新次数（默认 100，20=旧的一步一跳）和速度%
python policy_link.py once --observe-only --cycles 3  # 连服务器只发观测、打印回来的 chunk，臂不动；验证两路图像 + 16 维 state
python _can_listen.py           # 只收不发的 CAN 抓包，日志写到 _can_listen_log.txt
python _realsense_test.py       # 存两路 D405 的彩色图（原图 + 224×224 模型输入）到 _camera_test/，图上标角色和序列号
python check_candidate_repo.py PATH_TO_REPO   # 换模型前检查候选策略仓库
```

库调用 `read_state()`、`driver_faults()` 不发运动命令。

## 架构（跨文件才能看清的部分）

- `team_config.py` 是编排层：通过 SSH 在登录节点提交/复用名为 `serve-policy` 的 slurm 任务，轮询日志直到出现 `listening on`，用 `scontrol` 取 `NodeAddr`，在单独窗口（标题 `piper-link`）开 `ssh -L` 指向计算节点，然后调用 `policy_link.py once`。所有服务器路径、账号、checkpoint 只来自 `team_config.json`（该文件在 `.gitignore` 里，入库的是空模板 `team_config.example.json`）。缺字段时报出要填的键，绝不复用别人的路径。
- `policy_link.py` 持有整个循环的 WebSocket（与 `openpi_client` 线格式相同：先收一条 metadata，ndarray 用 `__ndarray__` + bytes 键 `data/dtype/shape` 编码）。每轮：`read_state()`（不完整则 raise 不发送）→ `PolicyCameras.read()`（两台 D405 各由常开的读取线程缓存最新帧，避免 pipeline 启动延迟；原图 pad 到 224×224）→ `build_observation()` → 等推理（最长 600 s，字符串回复即服务端异常）→ 把动作交给**常驻子进程** `policy_step.py worker`（`MotionWorker`，stdin/stdout 传 pickle，打印走 stderr）执行。读状态也经它。
- **相机与 CAN 写入不能在同一进程**（会以 `3221225477` access violation 退出）。`policy_link.py` 只持有相机，适配器只由运动子进程打开，每次运行只连一次（以前每轮重连+识别固件+0.5 s 停稳，两轮间停 1–3 s）。不要把这两部分合并回一个进程。
- `policy_step.py` 的 `action_waypoints()` 把 12 维输出解释为基座系下末端增量：0–2 平移 ×0.05 m，3–5 轴角 ×0.5 rad 左乘，6 为夹爪（≥0.5 闭合到 0 mm，否则张开到 `gripper_max_mm()`），7–11 丢弃。每步按 0.05 s、速度 50% 发送 MOVE P 位姿。欧拉约定与 `piper_fk` 一致：`R = Rz·Ry·Rx`。
- 所有 CAN 通信都经过官方 `pyAgxArm`。`piper_client.PiperClient` 是它外面的薄封装（度/毫米 ↔ 弧度/米），对外接口 `read_state`/`driver_faults`/`gripper_max_mm`/`move_gripper`/`exit_drag_teach`/`enable_arm`/`move_joints`/`move_pose` 保持不变；状态名字用 `piper_client.py` 自己的表（`TEACHING` 等），保证录好的数据和 `dataset_export.py` 不变；超过 0.5 s 的反馈算缺失。`exit_drag_teach()` 用 pyAgxArm 的 `ArmMsgMotionCtrl` 经私有 `_send_msg` 发 `0x150`（没有公开接口）。
- `piper_can_bus.py`：`create_piper_arm()` 按平台选 CAN 后端并自动识别固件配置（先等反馈到达再 `get_firmware()`）。Linux 上用 python-can 的 `socketcan`/`can0`，`bring_up_can()` 在接口关闭时拉起到 1 Mbps（需要 root）；Windows/macOS 上用本仓库注册的 `piper_usb` 接口：`_WinUsbTransport`（ctypes 调 WinUSB，读写用独立 OVERLAPPED 事件，`PiperUsbBus` 用锁串行）或 `_LibUsbTransport`（pyusb，后台线程排空 IN 端点到 256 帧队列，否则约十几帧后写入 errno 60 超时，不要去掉）。`open_transport()` 发 gs_usb 控制传输启动 48 MHz / 1 Mbps 总线；接收时丢掉发送回显（`echo_id != 0xFFFFFFFF`）；关闭时**不发**总线停止控制传输（会挂起进程）。注册 `piper_usb` 要同时改 `BACKENDS` 和三个模块里的 `VALID_INTERFACES`。
- Linux 相机走 `pyrealsense2`，按 `team_config.json` 的序列号取流。macOS 上走 OpenCV 的 AVFoundation（`camera_backend`，不需要 sudo；pip 的 `pyrealsense2-macosx` 在 macOS 27 上段错误）：RealSense 序列号 → `camera_usb_serials` 里的 USB(ASIC) 序列号 → `ioreg` 的 locationID → 按 pyobjc 枚举的 uniqueID 排序得到 OpenCV 编号；必须是 1280×720，居中裁 4:3 再缩到 640×480，无深度。角色也可设成 `uvc:` + USB 厂商号产品号（如 Logitech C930e 的 `uvc:046d0843`，只限 AVFoundation，同型号只能插一台），跳过 1280×720 检查，同样裁到 640×480。主臂固件 2.0.7 不再发 `0x155`–`0x157`/`0x159`（改发 `0x1C0`–`0x1C3`，不解释），所以 `leader_command()` 为 None、`demo.py` 靠操作者确认主臂断电。`create_piper_arm()` 在没设 `PIPER_FIRMWARE` 时自己读 `piper_firmware`。IQ9075 上没有 pip，依赖是解压到 `.pylib` 的 aarch64 wheel（含一个占位 `sqlite3.py`），运行时 `PYTHONPATH=.pylib`。发运动帧前 `require_healthy_drivers()` 检查驱动器故障（关节顶到机械限位会过流失能，只能断电清除），所以 `home_deg` 不能设在限位上。
- `team_config.py` 的 `server_mode`：`simple` 在 `remote_code_dir` 跑 `simple_policy_server.py`、不申请 GPU；`openpi` 在 `openpi_dir` 跑 `scripts/serve_policy.py`。远端路径可以写 `~/`，由 `remote_quote()` 展开。非 Windows 上隧道是后台 `ssh -f -N -L`，`down` 用 `pkill` 关闭。
- 从折叠休息位姿发末端位姿目标会被机械臂逆解拒绝（`TARGET_EXCEEDS_LIMIT`，只有夹爪动）。服务器回复可带 `"action_space": "joints"`，动作为 `(N, 7)` 绝对关节角加夹爪闭合程度（0 张开、1 合到 0 mm，中间值部分闭合；回放取录制宽度减行程的 10% 以便夹紧，超过行程 75% 则完全张开），由 `apply_joint_actions()` 用 MOVE J（`0x155`–`0x157`）执行；每步内部按 `--freq`（默认 100 Hz）插值更新目标（只按 20 Hz 换目标会让 J1/J2 一卡一卡），速度取 `--speed` 或 `joint_speed`；`check_joint_chunk()` 限制每步不超过 3 度。`simple_policy_server.py` 默认 `--mode demo` 就走这条路。
- `server_mode` 有 `simple`/`replay`/`openpi` 三种，`--server` 临时覆盖，`mode_overrides[mode]` 覆盖 Slurm 等键。任务带 `--comment=MODE`，换模式前要 `down`。本仓库 16 维 state / `(50, 12)` 动作契约来自 RoboCasa pi0.5（openpi 配置 `pi05_pretrain_human300`）；`openpi_python` 可指向装好它的 conda 环境。回放服务器回复 `"done": True` 时客户端正常退出。
- 回放服务器把录制里超出 `JOINT_LIMITS_DEG` 的关节角截到限位（手能把 J5 拖过 70°）；夹爪行程取 `gripper_stroke_mm`（默认 70，100 mm 夹爪设 100），经 `--gripper-stroke-mm` 传给服务器。录制前和回放前都先到 `READY_DEG`，保证起点一致。
- 训练数据：`episode_recorder.py` 只读录制（相机 `PolicyCameras` + `read_state()`，`KeyReader` 跨平台读键），格式和导出规则见 `数据采集.md`。一次录制同时写 `images/NNNNNN.jpg`（第三视角）和 `images_wrist/NNNNNN.jpg`（腕部视角）；JPEG 编码与写盘在 `_ImageWriter` 线程里做，否则 exFAT 上一对图约 30 ms 会把循环拖到 15 Hz。`--leader` 录主从遥操作（`collection_mode: leader_follower`）：`leader_command()` 读主臂广播的 `0x155`–`0x157`/`0x159` 存进 `lead_*` 列（主臂固件 2.0.7 不发这些帧，全为 NaN），**只用于诊断**——它不进 state/action，也不做任何导出判断。`dataset_export.demonstrating()` 按模式判断示教样本，主从用 `FOLLOWING_MODES`（实测从臂跟随时是 `CAN_CTRL`）。整条录制的门禁看**从臂自身**：`follower_motion()` 的路程 < `MIN_PATH_MM`（10 mm）且转角 < `MIN_ROTATION_DEG`（5°）就拒绝导出，主臂没上电和忘了开 teach 都能抓到，不需要主臂数据。主臂在线时两台臂的固件回复会交错，原始帧里第一个版本号可能是主臂的，所以 `create_piper_arm()` 不猜，要设 `PIPER_FIRMWARE` 或 `team_config.json` 的 `piper_firmware`。`_firmware_from_raw_bus` 只在单臂回答但 SDK 没返回时才用。回放服务器直接读这种 episode 文件夹（`episode_arrays()`），`setup-remote` 转成 `.npz` 上传——只含从臂的关节角和夹爪。
- `policy_step.check_limits()` 在发送任何帧前检查 `max_chunk_shift_mm`（离 chunk 起点的最大距离）和 `workspace_mm`（基座系绝对盒子），超限抛 `ChunkRefused`（运动进程回复 `refused`），`policy_link.py` 打印 `chunk_refused` 跳过这个 chunk 继续观测。不要绕过它。`execute_steps`/`ee_delta_scale` 只执行 chunk 前 N 步并缩放（仿真模型的整段 chunk 积分是米级）。

## 模型相关的契约（换模型时首先要改的地方）

observation 和 action 的含义随加载的 checkpoint 变化，不是本仓库的固定标准：
- 输入：`build_observation()` 与 `piper_train_state()`（16 维 state 的顺序硬编码，与 `prepare_dataset.piper_train_state()` 一致；发 `observation/image` 与 `observation/wrist_image` 两路）。`prompt` 是 `policy_link.py` 的 `PROMPT` 常量，当前是 `put the pen into the box`，换任务时改它和 `meta/tasks.jsonl`。
- 输出：`policy_step.py` 的 `action_waypoints()`。`simple_policy_server.py` 生成的动作也依赖这个含义。
- 换模型前先跑 `check_candidate_repo.py` 并读它指出的文件，弄清动作类型/单位/相机字段之前，不要把输出写到 `0x155`（关节）或 `0x159`。**不要把 12 维输出当作 6 个关节角写到 `0x155`。**

## 硬性约束

- `0x159` 夹爪帧的 set-zero 字节必须为 `0x00`；`0xAE` 会把当前位置写成零点，绝不能发送。不要为开启周期反馈去猜 `0x477` 的 byte 3。
- 不要回应 `0x710`；`0x1C0`–`0x1C3`、`0x212` 不做解释。
- 不要改回自己拼 CAN 帧，也不要用 `piper_sdk`、`piper_ros`、python-can 自带的 gs_usb、`pyAgxArm` 的 `agx_cando`：都不适用于这个适配器。pyAgxArm 一律经 `piper_can_bus.create_piper_arm()` 创建。`piper_usb` 在 macOS 实机上验证过读取，Windows 实机还没测。
- 绝不调用 pyAgxArm 的 `calibrate_gripper()` / `calibrate_joint()`（发 `0xAE`，写零点）。运动前保留 `policy_step.py` 的检查。
- 打开设备必须带 `FILE_FLAG_OVERLAPPED`，否则 `WinUsb_Initialize` 返回错误 6。错误 5 = ArmRobot/VMware USB 服务/残留 Python 进程占用了适配器。
- 臂处于 `TEACHING` 且 teach=1 时末端位姿命令会被忽略，需先 `exit_drag_teach()`（`0x150`，不是急停）。
- 文档和新代码里不要写任何人的桌面绝对路径；路径相对于本目录，服务器路径来自 `team_config.json`。
- 不要按深度给相机分配角色。`_camera_test/`、`episodes/`、`datasets/` 都在 `.gitignore` 里，抓下来的图和录制的数据永远不提交。
- 读到的关节角等现场数值不要拷进共享配置。
