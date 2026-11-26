#爆管响应矩阵构建
import wntr
import numpy as np
import os
import copy

inp_file=r"D:\A_科研相关\小论文3_动态监测网络\基于GNN的爆管响应预测\L-town\L-TOWN.inp"
wn=wntr.network.WaterNetworkModel(inp_file)
nodes=list(wn.node_name_list)
pipes=list(wn.pipe_name_list)
n_node,n_pipe=len(nodes),len(pipes)
node_id={n:i for i,n in enumerate(nodes)}

# 正常状态模拟
result=wntr.sim.EpanetSimulator(wn).run_sim()
normal_pressure=np.array([result.node["pressure"].iloc[0][n] for n in nodes])


# 爆管响应
response=np.zeros((n_node,n_pipe),dtype=np.float32)

for j,p in enumerate(pipes):
    print(f"burst {j+1}/{n_pipe}")
    wn_burst=copy.deepcopy(wn)
    pipe=wn_burst.get_link(p)
    pipe.initial_status="Closed"
    start=wn_burst.get_node(pipe.start_node_name)

    if start.node_type=="Junction":
        leak_node=start
    else:
        leak_node=wn_burst.get_node(pipe.end_node_name)
    leak_node.add_leak(wn_burst,area=burst_area)

    try:
        result_b=wntr.sim.EpanetSimulator(wn_burst).run_sim()
        burst_pressure=np.array([result_b.node["pressure"].iloc[0][n] for n in nodes])
        response[:,j]=normal_pressure-burst_pressure
    except Exception as e:
        print("failed:",p,e)