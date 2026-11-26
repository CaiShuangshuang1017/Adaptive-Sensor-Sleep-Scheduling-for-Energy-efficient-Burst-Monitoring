import numpy as np,pandas as pd,random,os

physical_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\improved_physical_schedule_result.xlsx"
coverage_output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\improved_physical_schedule_coverage_matrix.xlsx"

initial_energy=100
consume=1
coverage_ratio=1
save_interval=1

def calculate_physical_redundancy(coverage_matrix,active_sensors):
    G=coverage_matrix.loc[:,active_sensors].astype(bool);redundancy={}
    for sensor in active_sensors:
        sensor_cover=G[sensor];sensor_cover_num=int(sensor_cover.sum())
        if sensor_cover_num==0:
            redundancy[sensor]=1.0;continue
        others=[s for s in active_sensors if s!=sensor]
        other_cover=G[others].any(axis=1) if others else pd.Series(False,index=G.index)
        redundancy[sensor]=float((sensor_cover&other_cover).sum()/sensor_cover_num)
    return redundancy

def calculate_coverage(coverage_matrix,active_sensors):
    if not active_sensors: return 0,0
    covered=coverage_matrix.loc[:,active_sensors].astype(bool).any(axis=1)
    covered_num=int(covered.sum());coverage_rate=covered_num/len(coverage_matrix)*100 if len(coverage_matrix)>0 else 0
    return covered_num,coverage_rate

def physical_sensor_scheduling(sensor_ids,coverage_matrix,remaining_energy):
    active_sensors=sensor_ids.copy();test_sensors=sensor_ids.copy();each_iter_result={}
    base_covered_num,base_coverage_rate=calculate_coverage(coverage_matrix,active_sensors)
    required_num=int(np.ceil(base_covered_num*coverage_ratio));required_rate=required_num/len(coverage_matrix)*100 if len(coverage_matrix)>0 else 0
    print("全部传感器参考覆盖管段数:",base_covered_num);print("全部传感器参考覆盖率:",round(base_coverage_rate,2),"%");print("性能保持比例:",coverage_ratio);print("最低要求覆盖管段数:",required_num);print("最低要求覆盖率:",round(required_rate,2),"%")
    iteration=0
    while test_sensors:
        iteration+=1;redundancy=calculate_physical_redundancy(coverage_matrix,active_sensors)
        random_order={s:random.random() for s in test_sensors}
        candidates=sorted(test_sensors,key=lambda s:(remaining_energy[s],-redundancy.get(s,0),random_order[s]));sensor=candidates[0]
        print(f"\n******** 第 {iteration} 轮 ********");print(f"测试:{sensor} | 电量:{remaining_energy[sensor]:.2f} | 冗余度:{redundancy.get(sensor,0):.3f}")
        test_active=[s for s in active_sensors if s!=sensor]
        test_covered_num,test_coverage_rate=calculate_coverage(coverage_matrix,test_active)
        sensors_before=active_sensors.copy()
        if test_covered_num>=required_num:
            active_sensors=test_active;status="sleep"
            print(f"休眠成功:{sensor} | 覆盖率:{test_coverage_rate:.2f}%")
        else:
            status="retain"
            print(f"休眠失败:{sensor} | 覆盖率:{test_coverage_rate:.2f}% → 保持激活")
        test_sensors.remove(sensor)
        each_iter_result[iteration]={"sensor":sensor,"decision":status,"energy":remaining_energy[sensor],"redundancy":redundancy.get(sensor,0),"before_sensors":sensors_before,"after_sensors":active_sensors.copy(),"test_covered_num":test_covered_num,"test_coverage_rate":test_coverage_rate}
    final_covered_num,final_coverage_rate=calculate_coverage(coverage_matrix,active_sensors)
    final_redundancy=calculate_physical_redundancy(coverage_matrix,active_sensors)
    return active_sensors,final_redundancy,final_coverage_rate,each_iter_result,base_coverage_rate,base_covered_num,required_num

def save_results(results,coverage_results,output_file,coverage_output_file):
    if results: pd.DataFrame(results).to_excel(output_file,sheet_name="调度结果",index=False)
    if coverage_results:
        with pd.ExcelWriter(coverage_output_file,engine="openpyxl",mode="w") as writer:
            for key,G in coverage_results.items(): G.to_excel(writer,sheet_name=key[:31])
    print(f">>> 已保存累计 {len(results)} 个调度周期结果")

if __name__=="__main__":
    xls=pd.ExcelFile(physical_file);sheet_names=xls.sheet_names[:24]
    df0=pd.read_excel(physical_file,sheet_name=sheet_names[0],index_col=0)
    sensor_ids=list(df0.columns);n_sensor=len(sensor_ids);sensor_index={s:i for i,s in enumerate(sensor_ids)}

    all_sensor_energy={s:initial_energy for s in sensor_ids}
    results=[];coverage_results={};system_failure_time=None;depleted_final=[];global_step=0;cycle=0;stop_system=False

    try:
        while not stop_system:
            cycle+=1
            print("\n"+"="*80);print(f"开始第 {cycle} 个24状态循环");print("="*80)

            for state_idx,sheet in enumerate(sheet_names,1):
                global_step+=1
                print("\n"+"#"*70);print(f"全局时刻:{global_step} | 周期:{cycle} | 状态:{state_idx}/24 | sheet:{sheet}");print("#"*70)

                coverage_matrix=pd.read_excel(physical_file,sheet_name=sheet,index_col=0)
                coverage_matrix=coverage_matrix.loc[:,sensor_ids].fillna(0)
                temp_energy={s:all_sensor_energy[s] for s in sensor_ids}

                result=physical_sensor_scheduling(sensor_ids,coverage_matrix,temp_energy)
                final_active_ids,redundancy,coverage_rate,each_iter_result,ref_rate,ref_covered_num,required_num=result

                final_x=np.zeros(n_sensor,dtype=int)
                for sensor in final_active_ids: final_x[sensor_index[sensor]]=1

                # 保存当前调度后覆盖矩阵
                final_coverage_matrix=coverage_matrix.loc[:,final_active_ids].T.copy()
                coverage_key=f"t{global_step}_c{cycle}_s{state_idx}"
                coverage_results[coverage_key]=final_coverage_matrix

                for sensor in final_active_ids: all_sensor_energy[sensor]=max(all_sensor_energy[sensor]-consume,0)

                depleted_sensors=[s for s in sensor_ids if all_sensor_energy[s]<=0]
                energy_failure=len(depleted_sensors)>0
                if energy_failure:
                    system_failure_time=global_step;depleted_final=depleted_sensors.copy()

                final_state={s:[int(final_x[sensor_index[s]]),round(all_sensor_energy[s],2)] for s in sensor_ids}
                active_energy=[all_sensor_energy[s] for s in final_active_ids]
                mean_energy=round(np.mean(active_energy),2) if active_energy else np.nan
                all_energy=list(all_sensor_energy.values())
                energy_range=round(max(all_energy)-min(all_energy),2)
                energy_std=round(float(np.std(all_energy)),3)
                min_energy=round(min(all_energy),2);max_energy=round(max(all_energy),2)

                results.append({
                    "step":global_step,
                    "cycle":cycle,
                    "state_index":state_idx,
                    "hydraulic_state":sheet,
                    "active_sensor":",".join(map(str,final_active_ids)),
                    "active_number":len(final_active_ids),
                    "sleep_number":n_sensor-len(final_active_ids),
                    "active_ratio":len(final_active_ids)/n_sensor,
                    "active_mean_energy":mean_energy,
                    "energy_range":energy_range,
                    "energy_std":energy_std,
                    "min_energy":min_energy,
                    "max_energy":max_energy,
                    "coverage_rate":coverage_rate,
                    "reference_coverage_rate":ref_rate,
                    "reference_covered_num":ref_covered_num,
                    "required_covered_num":required_num,
                    "energy_failure":energy_failure,
                    "depleted_sensors":",".join(map(str,depleted_sensors)),
                    "redundancy":str(redundancy),
                    "iteration_result":str(each_iter_result),
                    "sensor_state":str(final_state)
                })

                print("\n---------------- 当前时刻最终结果 ----------------")
                print("水力状态:",sheet)
                print("全部传感器参考覆盖率:",round(ref_rate,2),"%")
                print("参考覆盖管段数:",ref_covered_num)
                print("最低要求覆盖管段数:",required_num)
                print("最终激活传感器:",final_active_ids)
                print("激活数量:",len(final_active_ids))
                print("休眠数量:",n_sensor-len(final_active_ids))
                print("优化后覆盖率:",round(coverage_rate,2),"%")
                print("最低剩余电量:",min_energy)
                print("最高剩余电量:",max_energy)
                print("全网电量极差:",energy_range)
                print("全网电量标准差:",energy_std)

                if len(results)%save_interval==0:
                    save_results(results,coverage_results,output_file,coverage_output_file)

                if energy_failure:
                    print("\n"+"!"*70)
                    print(f"监测寿命终止：第 {global_step} 个调度周期首次出现传感器电量归0")
                    print("电量耗尽传感器:",depleted_sensors)
                    print("!"*70)
                    save_results(results,coverage_results,output_file,coverage_output_file)
                    stop_system=True;break

    except Exception as e:
        print("\n程序异常:",e);print("正在保存已完成结果...")
        save_results(results,coverage_results,output_file,coverage_output_file)
        raise

    if results: save_results(results,coverage_results,output_file,coverage_output_file)

    print("\n"+"="*80)
    if system_failure_time is None:
        print("模拟结束，但未出现传感器电量归0。")
    else:
        print("监测寿命:",system_failure_time,"个调度周期")
        print("寿命终止循环:",cycle)
        print("首批电量耗尽传感器:",depleted_final)

    print("最终剩余电量:",all_sensor_energy)
    print("="*80)