仿真参数设置：utils.options.py

| 参数 | 类型 | 默认设置 | 含义 |
| --- | --- | --- | --- |
| task | str | “traj_pred” | 指定分布式训练的目标任务 |
| num_items | int | 1024 | 车辆本地数据集大小 |
| local_iter | int | 20 | 本地训练迭代次数 |
| local_bs | int | 64 | 本地训练的batchsize |
| lr | float | 0.01 | 学习率 |
| simple_eval | logic | False | 为减小模型验证的时间开销，使用原始验证集的一个子集（大小为simple_eval_num）作为验证集 |
| simple_eval_num | int | 1000 | 简单验证的验证集大小 |
| no_sumo_run | logic | False | 不运行SUMO仿真，直接采用现有的tripinfo.xml文件读取车流信息 |
| round_duration | float | 20 | 全局模型更新周期时长 |
| delay_download | float | 1 | 模型下发的通信时延 |
| delay_upload | float | 1 | 模型上传的通信时延 |
| mu_local_train | float | 0.1         | 本地训练计算时延分布参数mu                                   |
| beta_local_train | float | 0.1         | 本地训练计算时延分布参数beta                                 |
| Lambda          | float | 0.1         | 车辆到达率                                                   |
| maxSpeed | float | 20 | 车辆速度 |
| ckpt_path | str | “” | 若不为“”，则从ckpt_path加载checkpoint |
| non_iid          | logic | False       | 是否让车辆本地数据集non-i.i.d.                               |
| split_type | int | 0 | 指定non-i.i.d.的划分方式，0代表按城市non-i.i.d.，1代表按行为non-i.i.d. |
| city_skew | logic | False | 在模型聚合时是否对不同城市的车辆模型赋予不同加权系数 |
| behavior_skew | logic | False | 在模型聚合时是否对不同行为的车辆模型赋予不同加权系数 |
| skew | float | 0.5 | 在模型聚合时的加权系数偏置程度 |