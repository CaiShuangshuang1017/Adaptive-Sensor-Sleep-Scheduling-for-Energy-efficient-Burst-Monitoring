from pyscipopt import Model,quicksum
import pandas as pd
import numpy as np
import get_optimized_gain_coverage_matrix
from multiprocessing import freeze_support

input_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_schedule_result.xlsx"
covariance_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx"
burst_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
gain_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_coverage_matrix.xlsx"

initial_energy=100
consume=1
sensitivity_threshold=-4.22068293050037


if __name__=="__main__":
    freeze_support()

    excel=pd.ExcelFile(input_file)
    sheet_names=excel.sheet_names
    df0=pd.read_excel(input_file,sheet_name=sheet_names[0],index_col=0)

    sensor_ids=list(df0.columns)
    n_sensor=len(sensor_ids)
    sensor_index={s:i for i,s in enumerate(sensor_ids)}

    # 所有传感器真实剩余电量，始终保留所有传感器
    all_sensor_energy={sensor:initial_energy for sensor in sensor_ids}

    x_prev=np.zeros(n_sensor)
    gain_results=[]
    gain_coverage_results={}

    for sheet_idx,sheet in enumerate(sheet_names):
        covariance_matrix=pd.read_excel(covariance_file,sheet_name=sheet,index_col=0)
        pipe_burst_matrix=pd.read_excel(burst_file,sheet_name=sheet,index_col=0)

        # 当前优化使用电量副本，因为函数内部会 pop()
        temp_energy=all_sensor_energy.copy()

        final_active_ids,selected_combinations,gain_coverage_matrix,redundancy,coverage_rate,each_iter_result=\
            get_optimized_gain_coverage_matrix.iterative_remove_redundant_sensors(
                sensor_ids,
                covariance_matrix,
                pipe_burst_matrix,
                sensitivity_threshold,
                temp_energy
            )

        gain_coverage_results[sheet]=gain_coverage_matrix

        # 当前激活状态
        final_x=np.zeros(n_sensor)
        for sensor in final_active_ids:
            final_x[sensor_index[sensor]]=1

        # 只对最终激活传感器扣除一次能量
        for sensor in final_active_ids:
            all_sensor_energy[sensor]-=consume
            all_sensor_energy[sensor]=max(all_sensor_energy[sensor],0)

        # 保存当前所有传感器状态
        final_state={
            sensor:[
                int(final_x[sensor_index[sensor]]),
                round(all_sensor_energy[sensor],2)
            ]
            for sensor in sensor_ids
        }

        final_active_energy=[
            all_sensor_energy[s]
            for s in final_active_ids
        ]

        mean_energy=round(np.mean(final_active_energy),2) if final_active_energy else np.nan
        energy_range=round(max(final_active_energy)-min(final_active_energy),2) if final_active_energy else np.nan

        gain_results.append([
            sheet,
            final_state,
            ",".join(map(str,final_active_ids)),
            len(final_active_ids),
            mean_energy,
            energy_range,
            coverage_rate,
            selected_combinations,
            redundancy,
            each_iter_result
        ])

        # 更新上一时刻状态
        x_prev=final_x.copy()

        # 保存合作增益覆盖矩阵
        mode="w" if sheet_idx==0 else "a"
        with pd.ExcelWriter(
            gain_file,
            engine="openpyxl",
            mode=mode,
            **({} if mode=="w" else {"if_sheet_exists":"replace"})
        ) as writer:
            gain_coverage_matrix.to_excel(
                writer,
                sheet_name=str(sheet)[:31]
            )

        # 汇总优化结果
        gain_df=pd.DataFrame([
            {
                "time":r[0],
                **{str(k):v for k,v in r[1].items()},
                "active_sensor":r[2],
                "active_number":r[3],
                "mean_energy":r[4],
                "energy_range":r[5],
                "coverage_rate":r[6],
                "selected_combinations":str(r[7]),
                "redundancy":str(r[8]),
                "iteration_result":str(r[9])
            }
            for r in gain_results
        ])

        with pd.ExcelWriter(output_file,engine="openpyxl",mode="w") as writer:
            gain_df.to_excel(
                writer,
                sheet_name="优化结果",
                index=False
            )

        min_energy=min(all_sensor_energy.values())

        print(
            "时刻:",sheet,
            "激活传感器:",final_active_ids,
            "激活数量:",len(final_active_ids),
            "覆盖率:",coverage_rate,
            "最低剩余电量:",round(min_energy,2),
            "迭代过程:",each_iter_result
        )

        print(f"已保存 {sheet} 的优化结果")