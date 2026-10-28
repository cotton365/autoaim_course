# 数值后验：怎么跑、看什么

三条主验证线 + 协方差补充各自独立，可以并行跑。所有脚本只用本机 `/usr/bin/python3`
（numpy 1.21、scipy 1.8、matplotlib 3.5、cv2 4.5）。

```bash
cd /home/zh/RM/autoaim_course
export MPLCONFIGDIR=/tmp/mpl-zh

/usr/bin/python3 verify/verify_kf.py
/usr/bin/python3 verify/verify_pnp.py
/usr/bin/python3 verify/verify_projects.py
/usr/bin/python3 verify/verify_covariance.py
```

输出：

- `verify/kf_report.md`：一维 KF 等价性、2D 椭圆、NIS/NEES、EKF/IEKF/ESKF。
- `verify/pnp_report.md`：投影/PnP 闭环、深度误差、yaw 搜索、15° 俯仰、坐标系。
- `verify/projects_report.md`：三项目代码结论/疑点。
- `verify/covariance_report.md`：协方差 5 点手算、样本分母、`Cov(Ax)=A Aᵀ`。
- `verify/verify_results.json`：`verify_projects.py` 的机器可读结果。
- `assets/figures/*.png`：所有图。

## 结论速览（以脚本输出为准）

| # | 结论 | 证据 | 状态 |
|---|---|---|---|
| 1 | sp 用 `x[8]`（半径）判小陀螺，不是 `x[7]`（ω） | `proj_sp_index.png`；`r∈[0.05,0.5]`，阈值 2 恒真 | 确认 |
| 2 | sp 的 NIS 阈值 0.711 是 χ²₄ 的 5% 下分位 | `proj_nis_threshold.png`；`P(NIS>0.711)=0.95` | 确认 |
| 3 | sp 的 `min_x` 没有有效使用 | 3 次出现，2 次非注释，有效读取 0 次 | 确认 |
| 4 | SHtech 的 GEOMETRY 不是“错 180°”，而是同一 4 点集合的 id 重排 | `proj_shtech_geometry.png`；Hausdorff 0.62 mm，M≈-I | 修正 |
| 5 | SHtech 的 `armor_jump_interval` 被截成 0 | `proj_armor_jump.png`；200 帧中 gate 生效 0 次 | 确认 |
| 6 | SHtech 的 IESEKF 先验项符号与标准公式相反 | `proj_iekf_sign.png`；标量迭代 0.5/0 振荡，固定点 0.25 | 确认 |
| 7 | wuhan 的 `solve_pitch` 先返回高弧 p1 | `proj_wuhan_ballistic.png`；d=23.5 示例 p1=56.7°, t=5.08 s | 条件确认 |
| 8 | wuhan 的阻力前向模型自洽 | `forward_model` 残差 ~1e-14 m | 确认 |

> 第 4 条是这轮“后验”里最重要的修正：上一轮静态阅读说“SHtech 的 Planner 几何与 h() 差 180°”，
> 更准确的说法是 **同一 id 相差 180°、4 点集合完全相同**。按“最近板”选点时结果不受影响；
> `armor_index` 当前只赋值未读取，所以也没有实际的 id 影响。
>
> 第 7 条：RM 常见距离（<10 m）内 p1 通常超出 60° 被丢弃，所以这条更像“远距离/低初速的边界条件”，
> 不是日常必现的 bug。

## 图与报告

- KF：`kf_report.md`，图 `kf_prior_posterior.png`、`kf_cov_ellipse.png`、`kf_nis_nees.png`、`kf_iekf.png`、`kf_iekf_scalar.png`、`kf_eskf.png`。
- PnP：`pnp_report.md`，图 `pnp_projection.png`、`pnp_frames.png`、`pnp_yaw_scan.png`、`pnp_depth_noise.png`、`pnp_pitch_bias.png`。
- 项目：`projects_report.md`，图 `proj_*.png`。

## 复现约定

- 所有图用英文坐标轴标签，避免 matplotlib 中文字体缺失；中文解释在报告/课程里。
- PnP 的相机内参、装甲板尺寸、噪声水平在脚本开头集中定义，改一处即可重跑。
- 项目验证脚本只读三套仓库，不修改它们；所有输出写到本目录和 `assets/figures/`。
