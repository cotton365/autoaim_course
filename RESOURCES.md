# 自瞄预测课程 Resources

## Knowledge

- [Book: _Kalman and Bayesian Filters in Python_ — Roger Labbe](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
  从贝叶斯直觉讲到 EKF/UKF，代码可运行。用来看：KF 的 predict/update、协方差椭圆、NIS/NEES。
- [Paper: _Quaternion kinematics for the error-state Kalman filter_ — Solà](https://arxiv.org/abs/1711.02508)
  ESKF 的标准参考。用来看：nominal/error 分解、SO(3) 右扰动、inject/boxminus、误差状态传播。
- [Paper: _The iterated Kalman filter update as a Gauss-Newton method_ — Bell & Cathey (1993)](https://doi.org/10.1109/9.250523)
  IEKF 的原始出处。用来看：为什么迭代更新等价于高斯-牛顿，先验项从哪来。
- [Docs: OpenCV `solvePnP` / `solvePnPGeneric`](https://docs.opencv.org/4.x/d5/d1f/calib3d.html)
  PnP 的接口与 `SOLVEPNP_IPPE` 等旗标。用来看：输入输出、坐标系约定、重投影。
- [Paper: _Infinitesimal Plane-Based Pose Estimation_ — Collins & Bartoli (IJCV 2014)](https://doi.org/10.1007/s11263-014-0725-5)
  平面目标（装甲板）PnP 的经典方法，OpenCV 的 IPPE 实现来源。用来看：为什么平面 PnP 会有两个解、如何选正面。
- [Paper: _A Flexible New Technique for Camera Calibration_ — Zhang (TPAMI 2000)](https://doi.org/10.1109/34.888718)
  针孔模型、内参、畸变、标定的标准参考。用来看：K、D、像素坐标与归一化坐标。
- [Book: _Stochastic Models, Estimation, and Control_ — Maybeck](https://www.sciencedirect.com/book/9780124807015/stochastic-models-estimation-and-control)
  连续白噪声加速度模型、过程噪声离散化的经典推导。用来看：Q 里的 `dt^4/4, dt^3/2, dt^2` 从哪来。
- [Repo: sp_vision_25](https://github.com/)（本地 `/home/zh/RM/sp_vision_25`）
  11 维 EKF + PnP 球坐标观测；主参考实现。
- [Repo: wuhan/AWAKENING](https://github.com/WUST-RM/awakening)（本地 `/home/zh/RM/wuhan`）
  13 维 SO(3) ESKF + 灯条图像投影；附 `docs/algorithm/estimation.md`。
- [Repo: SHtech_auto_aim](https://github.com/)（本地 `/home/zh/RM/SHtech_auto_aim`）
  双模型 + IESEKF + 策略切换；工程化最完整。

## Wisdom (Communities)

- [RoboMaster 论坛](https://bbs.robomaster.com/)
  规则、兵种、视觉方案讨论。用来看：赛场约束、别人怎么处理遮挡/小陀螺。
- [OpenCV Forum](https://forum.opencv.org/)
  PnP/标定/坐标系问题的问答。用来看：`solvePnP` 坐标系、IPPE 双解等具体坑。
- 本地：RM 视觉组/战队代码 review
  用来看：参数为什么这么调、赛场上的失效模式。

## Gaps
- 没有一篇专门讲“RM 自瞄目标预测”的权威教材；三份代码本身是最重要的 primary source。
- 三项目都缺少完整的噪声标定文档；Q/R 多为经验值，需要用 NIS/NEES 自己验证。
