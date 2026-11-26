import pandas as pd

def check_structural_sensors_physical_all_states(original_coverage_file,output_file):
    excel=pd.ExcelFile(original_coverage_file)
    sheet_names=excel.sheet_names
    all_results=[]
    state_summary=[]

    for state_idx,sheet in enumerate(sheet_names,1):
        print("\n"+"="*80)
        print(f"水力状态 {state_idx}/{len(sheet_names)}: {sheet}")
        print("="*80)

        # 覆盖矩阵：行=管段，列=物理传感器，1/正值表示该传感器覆盖该管段
        coverage_matrix=pd.read_excel(original_coverage_file,sheet_name=sheet,index_col=0)
        sensor_ids=list(coverage_matrix.columns)

        # 原系统覆盖情况
        base_covered_mask=(coverage_matrix>0).any(axis=1)
        base_covered_num=int(base_covered_mask.sum())
        total_pipe_num=coverage_matrix.shape[0]
        base_rate=base_covered_num/total_pipe_num*100 if total_pipe_num>0 else 0

        print("管段总数:",total_pipe_num)
        print("原系统覆盖管段数:",base_covered_num)
        print("原系统覆盖率:",round(base_rate,2),"%")

        structural_sensors=[]
        removable_sensors=[]

        # ==================== 逐一删除传感器 ====================
        for idx,sensor in enumerate(sensor_ids,1):
            print(f"[{idx}/{len(sensor_ids)}] 删除传感器 {sensor}")

            test_matrix=coverage_matrix.drop(columns=[sensor])

            if test_matrix.shape[1]>0:
                test_covered_mask=(test_matrix>0).any(axis=1)
                test_covered_num=int(test_covered_mask.sum())
            else:
                test_covered_num=0

            test_rate=test_covered_num/total_pipe_num*100 if total_pipe_num>0 else 0

            # 覆盖损失
            coverage_loss=base_covered_num-test_covered_num
            coverage_rate_loss=base_rate-test_rate
            structural_impact=coverage_loss/base_covered_num if base_covered_num>0 else 0

            # 只要删除后覆盖管段减少，即认为结构性限制
            is_structural=coverage_loss>0

            if is_structural:
                structural_sensors.append(sensor)
                status="structural"
            else:
                removable_sensors.append(sensor)
                status="removable"

            all_results.append({
                "hydraulic_state":sheet,
                "state_index":state_idx,
                "sensor":sensor,
                "status":status,
                "structural":is_structural,
                "total_pipe_num":total_pipe_num,
                "base_covered_num":base_covered_num,
                "test_covered_num":test_covered_num,
                "coverage_loss":coverage_loss,
                "base_coverage_rate":base_rate,
                "test_coverage_rate":test_rate,
                "coverage_rate_loss":coverage_rate_loss,
                "structural_impact":structural_impact
            })

            print(
                f"{sensor}: {status} | "
                f"覆盖 {test_covered_num}/{base_covered_num} | "
                f"损失 {coverage_loss} 条管段 | "
                f"覆盖率下降 {coverage_rate_loss:.4f}%"
            )

        # ==================== 当前水力状态汇总 ====================
        state_summary.append({
            "hydraulic_state":sheet,
            "state_index":state_idx,
            "total_pipe_num":total_pipe_num,
            "base_covered_num":base_covered_num,
            "base_coverage_rate":base_rate,
            "structural_number":len(structural_sensors),
            "removable_number":len(removable_sensors),
            "structural_sensors":str(structural_sensors),
            "removable_sensors":str(removable_sensors)
        })

    # ==================== 汇总 ====================
    result_df=pd.DataFrame(all_results)
    summary_df=pd.DataFrame(state_summary)

    sensor_summary=result_df.groupby("sensor").agg(
        structural_times=("structural","sum"),
        mean_coverage_loss=("coverage_loss","mean"),
        max_coverage_loss=("coverage_loss","max"),
        mean_coverage_rate_loss=("coverage_rate_loss","mean"),
        max_coverage_rate_loss=("coverage_rate_loss","max"),
        mean_structural_impact=("structural_impact","mean"),
        max_structural_impact=("structural_impact","max")
    ).reset_index()

    sensor_summary["structural_ratio"]=sensor_summary["structural_times"]/len(sheet_names)
    sensor_summary["always_structural"]=sensor_summary["structural_times"]==len(sheet_names)
    sensor_summary["never_structural"]=sensor_summary["structural_times"]==0

    # ==================== 保存 ====================
    with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
        result_df.to_excel(writer,sheet_name="逐状态逐传感器",index=False)
        summary_df.to_excel(writer,sheet_name="各状态汇总",index=False)
        sensor_summary.to_excel(writer,sheet_name="传感器汇总",index=False)

    print("\n"+"="*80)
    print(f"{len(sheet_names)}个水力状态结构性限制分析完成")
    print("结果已保存:",output_file)
    print("="*80)

    return result_df,summary_df,sensor_summary


original_coverage_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_structural_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\structural_sensor_physical_24states.xlsx"

if __name__=="__main__":
    result_df,summary_df,sensor_summary=check_structural_sensors_physical_all_states(
        original_coverage_file,
        output_structural_file
    )