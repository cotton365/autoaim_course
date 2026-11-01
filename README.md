# 自瞄预测课：从 KF/PnP 到三套 RoboMaster 代码

这是一套“能落到代码变量名”的短课。目标不是背公式，而是让你能指着
`target.cpp` / `motion_model.hpp` / `Tracker.cpp` 里的某一行说：
**这个矩阵在算什么、这个噪声为什么这么设、这个时间戳为什么这样外推。**

> 主报告（先看这个）：[`三项目自瞄预测对照.md`](./三项目自瞄预测对照.md)
>
> 数值后验（所有结论的证据）：[`verify/`](./verify/)
>
> 速查卡：[`reference/`](./reference/)

---

## 1. 学习路径（7 个主课 + 1 个补充，每节 25–35 分钟）

| 顺序 | 课 | 你会得到什么 | 先修 |
|---|---|---|---|
| 1 | [KF：从高斯乘法到 predict/update](./lessons/0001-kf-from-gaussian.html) | 能手算一维 KF，理解 `P/F/Q/H/R/K` | 概率密度、矩阵乘法 |
| 1b | [补充：协方差：定义、矩阵、椭圆](./lessons/0001b-covariance.html) | 能说清 `P=E[(x−μ)(x−μ)ᵀ]`、椭圆、`A P Aᵀ` | 高中期望/方差 |
| 2 | [PnP：从四个像素到世界坐标](./lessons/0002-pnp-and-frames.html) | 能解释 `solvePnP`、重投影、yaw 搜索、15° 俯仰 | 针孔模型、旋转矩阵 |
| 3 | [噪声：Q/R、马氏距离、NIS](./lessons/0003-noise-covariance-nis.html) | 用导航例子算 `K/P⁺`，解释 Q/R、NIS 与自瞄误差 | [0001b 协方差](./lessons/0001b-covariance.html) |
| 4 | [机械与观测：从四角到车状态](./lessons/0004-three-pipelines.html) | 能说清单板 PnP 的 6DoF、车心/半径的可观测性、关联与初始化 | 2 |
| 5 | [预测：状态怎么走到 t_hit](./lessons/0005-ekf-eskf-iekf.html) | 能写状态转移、解释 EKF/ESKF/IEKF 各修什么故障 | 1、3、4 |
| 6 | [时间与开火：命中时刻不是开火时刻](./lessons/0006-ballistic-and-time.html) | 能推导弹道、解释飞行时间不动点、开火窗口 | 2、5 |
| 7 | [多板决策：打哪一块，先打哪一块](./lessons/0007-plate-selection.html) | 能定义 `p_hit/t_aim/t_window`，写出选板分数与门限 | 5、6 |

每节课的结构固定：**一个可验收的小目标 → 几何/概率直觉 → 公式 → 代码位置 → 数值例子 → 3 道检索题**。
题目答案不在页面上直接给出，点选项会立即反馈。

4–7 按同一根链走：**观测层 → 预测层 → 时间/开火层 → 决策层**。每一层都先问“这一层要解决什么问题”，再看三套代码各自怎么应对；不按项目逐个讲。

---

## 2. 三条并行线

这个工作区是按“并行”组织的：

```text
线 A：基础课        lessons/0001–0003
线 B：四层链路      lessons/0004–0007 + 三项目自瞄预测对照.md
线 C：数值后验      verify/ + assets/figures/
```

三条线互相引用：基础课里的公式，会在项目对照里标出对应的代码行；
项目对照里的每个“结论/疑点”，尽量在 `verify/` 里有对应的数值实验。

---

## 3. 文件地图

```text
autoaim_course/
├── README.md                          ← 你在这里
├── MISSION.md                         ← 为什么学、学成什么样
├── NOTES.md                           ← 教学偏好/环境笔记
├── RESOURCES.md                       ← 可信资料清单
├── 三项目自瞄预测对照.md               ← 主报告（Markdown）
├── lessons/
│   ├── 0001-kf-from-gaussian.html
│   ├── 0001b-covariance.html
│   ├── 0002-pnp-and-frames.html
│   ├── 0003-noise-covariance-nis.html
│   ├── 0004-three-pipelines.html
│   ├── 0005-ekf-eskf-iekf.html
│   ├── 0006-ballistic-and-time.html
│   └── 0007-plate-selection.html
├── reference/
│   ├── kf-cheatsheet.html
│   ├── pnp-cheatsheet.html
│   ├── noise-cheatsheet.html
│   ├── project-map.html
│   └── glossary.html
├── assets/
│   ├── course.css
│   ├── quiz.js
│   └── figures/*.png
├── learning-records/
│   ├── 0001-initial-gaps.md
│   └── 0002-high-school-probability.md
└── verify/
    ├── verify_kf.py / kf_report.md
    ├── verify_pnp.py / pnp_report.md
    ├── verify_projects.py / projects_report.md
    └── verify_covariance.py / covariance_report.md
```

---

## 4. 怎么“后验”自己学会了

每节课末尾有一道“能不能用一句话解释”的检索题。真正的后验在 `verify/`：

- 跑 `MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_kf.py`，看 KF 后验均值是否等于高斯乘法、NIS 是否落在 χ² 区间；
- 跑 `MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_pnp.py`，看 PnP 反解误差、yaw 搜索最小值、15° 俯仰偏差；
- 跑 `MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_projects.py`，看三项目里的“疑点”到底成不成立；
- 跑 `MPLCONFIGDIR=/tmp/mpl-zh /usr/bin/python3 verify/verify_covariance.py`，看协方差 5 点手算和 `Cov(Ax)=A Aᵀ`。

图都在 `assets/figures/`。如果你发现图和正文对不上，以图和脚本为准，回来改正文。

---

## 5. 开课方式（关于 skill）

这个会话的技能目录里没有教学类技能；本机 `~/.agents/skills/teach/SKILL.md` 存在，但被标记为
`disable-model-invocation: true`，不能通过 `skill` 工具调用。所以这里**按 teach skill 的工作区规范手工开课**：
`MISSION.md` / `RESOURCES.md` / `NOTES.md` / `learning-records/` / `lessons/` / `reference/` / `assets/` 都按它的结构建。
你可以把 `autoaim_course/` 直接注册成一个教学 workspace 继续迭代。

---

## 6. 一句话地图

```text
像素 → PnP/投影 → 观测 z
                 ↓
状态 x = [车心, 速度, 朝向, 角速度, 半径/高度]
                 ↓
EKF / ESKF / IEKF：predict + update
                 ↓
外推到 t_hit = t_now + latency + T_fly
                 ↓
选板 → 弹道 pitch → 云台 yaw/pitch → 开火
```
