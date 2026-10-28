---
name: autoaim-prediction-course
description: 教用户读懂 RoboMaster 三套自瞄（sp_vision_25 / wuhan / SHtech）的目标预测数学与代码；包含 KF/PnP/噪声基础、三项目对照、数值后验。
---

# 自瞄预测课（Teaching Workspace）

这是一个按 `teach` skill 规范搭建的教学 workspace，主题是“RoboMaster 自瞄的目标预测”。
用户的目标、成功标准、约束写在 `MISSION.md`；教学偏好写在 `NOTES.md`；可信资料在 `RESOURCES.md`。

## 什么时候用

- 用户问“KF/PnP/ESKF/IEKF 在这三套代码里到底怎么用”；
- 用户要对照 `sp_vision_25`、`wuhan`、`SHtech_auto_aim` 的预测链路；
- 用户要复现/检验某个结论（索引、NIS 阈值、IEKF 符号、弹道高弧等）。

## 怎么教

1. 先读 `MISSION.md` 和 `learning-records/`，确认用户当前的基础与目标。
2. 按 `lessons/0001` → `0001b` → `0006` 的顺序推进；每节课只解决一个可验收的小目标。
3. 公式必须落到代码变量名（文件:行）；每个结论尽量配 `verify/` 里的脚本/图。
4. 不要直接给答案；先让用户预测/手算，再用图或数值反馈。
5. 如果用户提出新问题，先判断它属于哪条线：
   - 基础线 → 补 `lessons/0001–0003`；
   - 项目线 → 查 `三项目自瞄预测对照.md` 和 `reference/project-map.html`；
   - 验证线 → 跑 `verify/verify_*.py`，用图说话。

## 关键文件

- `index.html`：课程首页（浏览器打开）。
- `三项目自瞄预测对照.md`：主报告。
- `lessons/0001–0006.html` + `0001b-covariance.html`：六节课 + 协方差补充。
- `reference/*.html`：速查卡。
- `verify/*.py`、`verify/*_report.md`：数值后验。
- `assets/figures/*.png`：所有图。

## 已知的“坑”与结论

- sp：`choose_aim_point` 用 `x[8]`（半径）而不是 `x[7]`（ω）判小陀螺；NIS 阈值 0.711 是 χ²₄ 的 5% 下分位。
- SHtech：`armor_jump_interval` 被 `int` 截成 0；IESEKF 先验项符号与标准公式相反；`predict_closest_armor` 是同一 4 点集合的 id 重排（`p_i=h_{i+2}`），`armor_index` 未读取。
- wuhan：`solve_pitch` 先返回高弧 p1；RM 常见距离内通常 p1 超出 60° 被丢弃。
- PnP：平面 IPPE 双解可能都“正面朝相机”，单靠法线筛不掉；需要 +15° pitch 先验/时域一致性。

## 环境

- 绘图/验证：`/usr/bin/python3`（matplotlib 3.5、scipy 1.8、cv2 4.5）。
- `MPLCONFIGDIR=/tmp/mpl-zh`。
- 三套仓库只读；所有输出写到本 workspace。
