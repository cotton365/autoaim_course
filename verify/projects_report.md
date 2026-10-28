# 三项目代码结论的数值后验

脚本：`verify/verify_projects.py`；原始输出：`verify/verify_output.txt`；
机器可读结果：`verify/verify_results.json`；图：`assets/figures/proj_*.png`。

## 1. sp_vision_25：`choose_aim_point` 的 `x[8]` vs `x[7]`

代码事实：

- `target.cpp:34` 注释与 `:39` 初始化确认状态顺序 `[x,vx,y,vy,z,vz,a,w,r,l,h]`；
- `target.cpp:243-244` 的 `diverged()` 把 `x[8]` 当半径 `r`，并限制 `0.05 < r < 0.5`；
- `aimer.cpp:163` 却写 `std::abs(target.ekf_x()[8]) <= 2` 来区分“非小陀螺”；
- `aimer.cpp:41/204` 判断角速度时用的是 `x[7]`。

数值：

```text
r samples [0.05, 0.1, 0.2, 0.26, 0.5] → |x[8]|<=2 全部 True
w samples [0, 1, 1.99, 2, 2.51, 4, 6.28, 10] → |x[7]|<=2: T T T T F F F F
threshold 2 / r_max 0.5 = 4
```

结论：**确认**。对非前哨站，`|r|<=2` 恒真，`coming/leaving` 分支实际上不会走到；
如果本意是判断小陀螺，应读 `x[7]`。

图：`proj_sp_index.png`。

另外：`tracker.cpp:271` 的 `min_x` 出现 3 次，2 次非注释，**有效读取 0 次**；
`|| armor.center.x != min_x` 被注释掉，所以“只更新最左板”的逻辑没有生效。

## 2. sp_vision_25：NIS 阈值 0.711

代码事实：`extended_kalman_filter.cpp:66` 写 `constexpr double nis_threshold = 0.711;`，
注释是“自由度=4，置信水平95%”；`tracker.cpp:86` 的门限是 `0.4 * window_size = 40`，
`window_size=100`。

数值：

```text
χ²₄ 5% 下分位  = 0.710723
χ²₄ 95% 上分位 = 9.487729
0.711 的分位   = 5.003%
P(NIS > 0.711 | 一致滤波器) = 0.94997
100 帧中期望失败 ≈ 95 次
P(100 帧中失败 < 40 次 | 一致滤波器) ≈ 5.7e-53
```

结论：**确认**。0.711 是 5% 下分位，不是 95% 上分位；若滤波器一致，tracker 的 40/100
门限几乎必然触发“Bad Converge Found”。实际能稳定跟踪，说明 `R` 被显著放大或 NIS 长期偏小，
但注释与数值肯定不符。

补充：代码算的并不是标准 pre-fit NIS——`P`、`x` 已经先更新，`residual` 是 post-fit，
`S = H P_post H' + R`；在 `H=I, P=I, R=I` 的各向同性例子里，代码 NIS 约为 `χ²₄/3`，
`P(NIS_code > 0.711) = 0.7113`。所以上面的 χ²₄ 数字是“若它是标准 NIS”的参照，真实分布
必须用录包数据确认。另外 `nees_threshold` 也是 0.711，但状态维数是 11；χ²₁₁ 的 95% 上分位
是 19.675，0.711 在 χ²₁₁ 上只有 0.087% 分位。

图：`proj_nis_threshold.png`。

## 3. SHtech：`h_armor_xyz()` vs `predict_closest_armor()`

先澄清坐标：`h_armor_xyz()` 返回 `{y_a, x_a, z_a}`，物理坐标是

```text
x_a = x_c - r sin(θ_i)
y_a = y_c - r cos(θ_i)
```

Planner 用

```text
x = x_c + r sin(θ_i)
y = y_c + r cos(θ_i)
```

数值（`c=(0,3)`，`r=0.26`，`l=h=0`）：

| yaw | 同一 id 距离 | 4 点集合 Hausdorff | 拟合映射 |
|---:|---:|---:|---|
| 0.0 | 0.520000 m | 0.62 mm（1.57）/ 3e-17（π/2） | `M≈-I`，180° |
| 0.7 | 0.520000 m | 0.62 mm（1.57）/ 4e-16（π/2） | `M≈-I`，180° |

`planner id -> nearest h id` 都是 `[2,3,0,1]`。

结论：**修正上一轮的过强说法**。不是“Planner 几何整体错 180°”，而是：

1. 同一 `id` 的两套点相差 180°（`p_i = h_{i+2}` 的索引重排）；
2. **4 个点的集合完全相同**（Hausdorff 0.62 mm，来自 `1.57` 而不是 `π/2`）；
3. 因为 Planner 选的是“离原点最近的板”，集合相同 → 选点结果不受影响；
4. `armor_index` 目前只赋值、未读取（`Planner.cpp:219/221/255`），所以当前瞄准**数值**也不受影响；
   如果未来用 `armor_index` 做开火/跟踪记忆，才需要注意 `p_i=h_{i+2}` 的 id 重排。

图：`proj_shtech_geometry.png`。

## 4. SHtech：`armor_jump_interval` 被截成 0

代码事实：`Planner.hpp:80` 是 `const int armor_jump_interval = 0.1f;`；
`Planner.cpp:262` 是 `if (dt_since_jump > armor_jump_interval)`。

C++ 输出：

```text
float_0.1f_as_double      = 0.10000000149
armor_jump_interval_as_int = 0
dt_zero_gt_interval        = 0
armor_jump_at_dt_zero      = 1
post_jump_frames           = 200
armor_jump_true_frames     = 0
intended_true_frames       = 100
```

结论：**确认**。门控宽度为 0；`dt>0` 时 `armor_jump` 恒为 false，
“换板后 0.1 s 内禁火”实际没有生效。

图：`proj_armor_jump.png`。

## 5. SHtech：IESEKF 先验项符号

代码事实（`IESEKF_Double_Armor.hpp:243-251`）：

```text
dx_pri = x_minus(X_pri, X_cur)      // X_pri - X_cur
dx     = K * (innovation + H * dx_pri)
X_next = x_add(X_pri, dx)
```

标量 `x_pri=0, P=H=R=1, z=1`：

| iteration | 标准 IEKF | 代码符号 |
|---:|---:|---:|
| 0 | 0.0 | 0.0 |
| 1 | 0.5 | 0.5 |
| 2 | 0.5 | 0.0 |
| 3 | 0.5 | 0.5 |
| 4 | 0.5 | 0.0 |
| 5 | 0.5 | 0.5 |

代码映射 `x_{k+1}=0.5 - x_k`，导数 -1，固定点 0.25；5 次落点取决于奇偶。

结论：**确认**。标准 IEKF 是 `+H(x_k ⊖ x_pri)`；代码符号相反。
在真实 4D/11D 系统里不一定发散（K 小、阻尼大），但复现公式时不要照抄。

图：`proj_iekf_sign.png`。

## 6. wuhan：`solve_pitch` 的高弧优先

代码事实（`ballistic_trajectory.hpp:78-93`）：

```text
u1=(A+√Δ)/(2B), u2=(A-√Δ)/(2B)
if valid(p1) return p1;
if valid(p2) return p2;
```

构造 `k=0.092, g=9.8, v0=30, d=23.5, h=1`：

```text
A=83.5706, B=38.0243, C=2.7857, Δ=1048.57
p1=56.7407°, t1=5.0794 s
p2=33.9449°, t2=3.3580 s
returned = p1
```

前向模型回代残差：`|dx|=0`，`|dy|=1.42e-14 m`（p1/p2 都通过 1e-6 检查）。

结论：**条件确认**。两个解都有效时代码确实返回高弧；本次只验证了 `d=23.5, h=1, v0=30`
这一组远距离参数。近距离时 `p1` 可能超过 60° 被丢弃（未做全距离扫描，属于推断），
所以这是边界条件，不是日常必现。对比：sp 的真空弹道对同一场景取 `pitch=9.879°`、`t=0.795 s`（低弧）。

> 图 `proj_wuhan_ballistic.png` 已修正标注：残差取轨迹末端点，图中为 `|dx|=0, |dy|=1.4e-14 m`，
> 与 JSON 的 `forward_p1_abs_max_err_m=1.42e-14` 一致。旧版曾把 `t=0` 的点代入，出现
> `|dx|=2.4e+01, |dy|=1.0`，那是 annotation bug，不是弹道误差。

## 7. 只做了静态检查、没有数值实验的点

- SHtech `cal_gimbal_target` 用 `atan(pi(0)/pi(2))` 而不是 `atan2`：目标在后方会丢象限；
  这是代码事实，是否触发取决于实际角度范围。
- SHtech `robot_speed_mps` 实际是弹速（`uart_driver.cpp:76` 把 `shoot_speed` 赋给它），
  README 也说明当前是 mock `24.5 m/s`。
- wuhan `R` 里的 `σ²/2`：可能是“两端点平均后方差减半”的考虑，但角度/长度不是平均量；
  没有做实验，保留为待核对。
- sp `update_target()` 对同名同型的所有装甲板都 update；正常场景同名板唯一，未做多车实验。

## 一句话总结

这轮数值后验**确认**了 sp 的索引/NIS、SHtech 的换板门控/IEKF 符号、wuhan 的高弧优先；
**修正**了“SHtech Planner 几何整体错 180°”这个过强说法——它是同一 4 点集合的 id 重排。
