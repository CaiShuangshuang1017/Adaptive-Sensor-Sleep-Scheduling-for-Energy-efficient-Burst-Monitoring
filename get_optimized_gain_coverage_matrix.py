import numpy as np,pandas as pd,random,os
from multiprocessing import Pool
from collections import Counter
import ast
from pyscipopt import Model,quicksum

pipe_burst_array=None
sensor_pool=None
cov_matrix_global=None
population_size_global=None
generations_global=None
mutation_rate_global=None

def initialize_population(population_size,current_physical_sensors,max_len=10):
    max_len=min(max_len,len(current_physical_sensors))
    return [random.sample(current_physical_sensors,random.randint(1,max_len)) for _ in range(population_size)]

def evaluate_population(population,covariance_matrix,burst_response):
    results=[]
    for individual in population:
        cov_sub=covariance_matrix.loc[individual,individual]
        std=np.sqrt(max(cov_sub.values.sum(),1e-10))
        decline=burst_response[individual].sum()
        results.append(decline/std)
    return results

def select(population,sensitivity_results):
    fitness=(np.max(sensitivity_results)-sensitivity_results)/(np.max(sensitivity_results)-np.min(sensitivity_results)+1e-10)
    probabilities=fitness/fitness.sum() if fitness.sum()>0 else np.ones(len(population))/len(population)
    return [population[i] for i in np.random.choice(len(population),len(population),p=probabilities)]

def crossover(parent1,parent2,max_len=10):
    union=list(set(parent1)|set(parent2))
    random.shuffle(union)
    max_len=min(max_len,len(union))
    return union[:random.randint(1,max_len)],union[:random.randint(1,max_len)]

def mutate(individual,mutation_rate,current_physical_sensors,max_len=10):
    individual=individual.copy()
    if random.random()<mutation_rate:
        mutation_type=random.choice(["add","delete","swap"])
        if mutation_type=="add":
            available=list(set(current_physical_sensors)-set(individual))
            if available and len(individual)<max_len:
                individual.append(random.choice(available))
        elif mutation_type=="delete" and len(individual)>1:
            individual.remove(random.choice(individual))
        elif mutation_type=="swap":
            available=list(set(current_physical_sensors)-set(individual))
            if available:
                idx=random.randrange(len(individual))
                individual[idx]=random.choice(available)
    return individual

def genetic_algorithm(covariance_matrix,burst_response,population_size,current_physical_sensors,generations,mutation_rate):
    population=initialize_population(population_size,current_physical_sensors)
    best=None
    best_value=float("inf")
    for g in range(generations):
        values=evaluate_population(population,covariance_matrix,burst_response)
        idx=np.argmin(values)
        if values[idx]<best_value:
            best_value=values[idx]
            best=population[idx].copy()
        selected=select(population,values)
        new_population=[]
        for i in range(0,len(selected)-1,2):
            c1,c2=crossover(selected[i],selected[i+1])
            new_population.extend([mutate(c1,mutation_rate,current_physical_sensors),mutate(c2,mutation_rate,current_physical_sensors)])
        population=new_population
    return best,best_value

def init_worker(pipe_array,sensors,cov_matrix,pop_size,generations,mutation_rate):
    global pipe_burst_array,sensor_pool,cov_matrix_global,population_size_global,generations_global,mutation_rate_global
    pipe_burst_array=pipe_array
    sensor_pool=sensors
    cov_matrix_global=cov_matrix
    population_size_global=pop_size
    generations_global=generations
    mutation_rate_global=mutation_rate
    np.random.seed(os.getpid())
    random.seed(os.getpid())

def optimize_single_pipe(pipe_index):
    burst_response=pd.Series(pipe_burst_array[pipe_index],index=sensor_pool)
    best_combination,max_sensitivity=genetic_algorithm(cov_matrix_global,burst_response,population_size_global,sensor_pool,generations_global,mutation_rate_global)
    return pipe_index,best_combination,max_sensitivity

def select_top_combinations_by_coverage(gain_coverage_matrix,current_physical_sensors,remaining_energy,required_num,consume,max_num=50):
    G=gain_coverage_matrix.astype(bool).copy(); combinations=list(G.index); n_pipe=G.shape[1]
    combination_sensor_dict={}
    for c in combinations:
        if isinstance(c,(tuple,list,set)): combination_sensor_dict[c]=list(c)
        elif isinstance(c,str): combination_sensor_dict[c]=list(ast.literal_eval(c))
        else: raise ValueError(f"无法解析组合: {c}")

    active_sensors=set(current_physical_sensors); removed_sensors=[]; current_combinations=combinations.copy(); current_G=G.copy()
    iteration=1; redundancy={}

    print("\n========== 理想点物理传感器删减 ==========")
    print("当前最大覆盖率:",round(G.any(axis=0).mean()*100,2),"%")
    print("固定最低覆盖率:",round(required_num/n_pipe*100,2),"%")

    while len(removed_sensors)<max_num:
        # ---------- 1. 重新计算当前合作覆盖冗余度 ----------
        redundancy={}
        for sensor in active_sensors:
            contain=[c for c in current_combinations if sensor in combination_sensor_dict[c]]
            noncontain=[c for c in current_combinations if sensor not in combination_sensor_dict[c]]

            if not contain:
                redundancy[sensor]=1.0
                continue

            contain_cover=current_G.loc[contain].any(axis=0)
            other_cover=current_G.loc[noncontain].any(axis=0) if noncontain else pd.Series(False,index=current_G.columns)
            contain_num=int(contain_cover.sum())
            redundancy[sensor]=float((contain_cover & other_cover).sum()/contain_num) if contain_num>0 else 1.0

        # ---------- 2. 找出休眠后仍满足覆盖约束的候选传感器 ----------
        feasible_candidates=[]
        for sensor in active_sensors:
            valid_combinations=[c for c in current_combinations if sensor not in combination_sensor_dict[c]]
            covered_num=int(current_G.loc[valid_combinations].any(axis=0).sum()) if valid_combinations else 0

            if covered_num<required_num:
                continue

            feasible_candidates.append({
                "sensor":sensor,
                "energy":remaining_energy[sensor],
                "redundancy":redundancy[sensor],
                "covered_num":covered_num,
                "coverage_rate":covered_num/n_pipe*100,
                "valid_combinations":valid_combinations
            })

        # ---------- 3. 无可休眠节点则结束 ----------
        if not feasible_candidates:
            print("当前已不存在休眠后仍满足覆盖约束的物理传感器，优化结束。")
            break

        # ---------- 4. 对全部可行候选进行归一化 ----------
        energies=np.array([x["energy"] for x in feasible_candidates],dtype=float)
        redundancies=np.array([x["redundancy"] for x in feasible_candidates],dtype=float)

        e_min,e_max=energies.min(),energies.max()
        r_min,r_max=redundancies.min(),redundancies.max()

        for x in feasible_candidates:
            # 电量越低越接近0
            x["energy_norm"]=(x["energy"]-e_min)/(e_max-e_min) if e_max>e_min else 0.0

            # 冗余度越高越接近1
            x["redundancy_norm"]=(x["redundancy"]-r_min)/(r_max-r_min) if r_max>r_min else 1.0

            # 理想休眠点：(0,1)
            x["ideal_distance"]=float(np.sqrt(
                x["energy_norm"]**2+
                (1-x["redundancy_norm"])**2
            ))

        # ---------- 5. 选择距离理想休眠点最近的节点 ----------
        # 距离相同时：低电量优先，再高冗余
        best=min(
            feasible_candidates,
            key=lambda x:(
                x["ideal_distance"],
                x["energy"],
                -x["redundancy"],
                str(x["sensor"])
            )
        )

        remove_sensor=best["sensor"]
        active_sensors.remove(remove_sensor)
        removed_sensors.append(remove_sensor)

        current_combinations=best["valid_combinations"]
        current_G=G.loc[current_combinations] if current_combinations else G.iloc[0:0]

        candidate_info=[
            (
                x["sensor"],
                round(x["energy"],2),
                round(x["redundancy"],3),
                round(x["ideal_distance"],3)
            )
            for x in sorted(feasible_candidates,key=lambda x:x["ideal_distance"])
        ]

        print(
            f"第 {iteration} 次删除 | 传感器:{remove_sensor} | "
            f"电量:{best['energy']:.2f} | "
            f"冗余度:{best['redundancy']:.3f} | "
            f"理想点距离:{best['ideal_distance']:.3f} | "
            f"覆盖率:{best['coverage_rate']:.2f}%"
        )
        print("候选(sensor,energy,redundancy,distance):",candidate_info)

        iteration+=1

    final_covered_num=int(current_G.any(axis=0).sum()) if len(current_G)>0 else 0
    final_coverage_rate=final_covered_num/n_pipe*100

    # ---------- 6. 最终能量均衡评价 ----------
    final_energy_after=[
        max(remaining_energy[s]-consume,0) if s in active_sensors else remaining_energy[s]
        for s in remaining_energy
    ]

    final_energy_std=float(np.std(final_energy_after))
    final_energy_range=float(max(final_energy_after)-min(final_energy_after))

    print("\n========== 最终删减结果 ==========")
    print("休眠物理传感器:",sorted(removed_sensors,key=str))
    print("激活物理传感器:",sorted(active_sensors,key=str))
    print("休眠数量:",len(removed_sensors))
    print("激活数量:",len(active_sensors))
    print("最终有效虚拟传感器数量:",len(current_combinations))
    print("最终覆盖率:",round(final_coverage_rate,2),"%")
    print("预测全网剩余电量标准差:",round(final_energy_std,3))
    print("预测全网剩余电量极差:",round(final_energy_range,2))

    return sorted(active_sensors,key=str),current_combinations,current_G,removed_sensors,redundancy,final_coverage_rate

def design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold):
    if not current_physical_sensors: return pd.DataFrame(columns=pipe_burst_matrix.index),0,{}
    pipe_array=pipe_burst_matrix.loc[:,current_physical_sensors].values; cov_matrix=covariance_matrix.loc[current_physical_sensors,current_physical_sensors]
    with Pool(processes=4,initializer=init_worker,initargs=(pipe_array,current_physical_sensors,cov_matrix,50,5,0.2)) as pool: results=pool.map(optimize_single_pipe,range(pipe_array.shape[0]))
    results.sort(key=lambda x:x[0]); all_pipe_best_combination=[r[1] for r in results if r[1]]
    unique_combinations=[]; seen=set()
    for combination in all_pipe_best_combination:
        key=tuple(sorted(combination))
        if key not in seen: unique_combinations.append(list(key)); seen.add(key)
    if not unique_combinations: return pd.DataFrame(columns=pipe_burst_matrix.index),0,{}
    sensor_index={s:i for i,s in enumerate(current_physical_sensors)}; individual_info=[]
    for individual in unique_combinations:
        idx=np.array([sensor_index[i] for i in individual]); cov_sub=covariance_matrix.loc[individual,individual].values
        std=np.sqrt(max(cov_sub.sum(),1e-10)); individual_info.append((idx,std))
    gain_coverage_dict=[]
    for pipe_index in range(pipe_array.shape[0]):
        pipe_response=pipe_array[pipe_index]; gain_sensitivity=[pipe_response[idx].sum()/std for idx,std in individual_info]
        gain_coverage_dict.append([1 if val<sensitivity_threshold else 0 for val in gain_sensitivity])
    combination_keys=[str(i) for i in unique_combinations]; combination_map={str(i):i for i in unique_combinations}
    gain_coverage_matrix=pd.DataFrame(np.array(gain_coverage_dict).T,index=combination_keys,columns=pipe_burst_matrix.index)
    coverage_rate=gain_coverage_matrix.any(axis=0).mean()*100
    return gain_coverage_matrix,coverage_rate,combination_map


def iterative_remove_redundant_sensors(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy,coverage_ratio=1,max_outer_iter=50):
    current_physical_sensors=current_physical_sensors.copy()
    each_iter_result={}
    gain_coverage_matrix,base_coverage_rate,combination_map=design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold)
    n_pipe=gain_coverage_matrix.shape[1];
    base_covered_num=int(gain_coverage_matrix.any(axis=0).sum())
    required_num=int(np.ceil(base_covered_num*coverage_ratio))
    print("\n================================================")
    print("全部传感器基准合作覆盖率:",round(base_coverage_rate,2),"%")
    print("基准覆盖管段数:",base_covered_num)
    print("最低要求覆盖管段数:",required_num)
    print("最低要求覆盖率:",round(required_num/n_pipe*100,2),"%")
    print("================================================")
    final_gain_matrix=gain_coverage_matrix
    final_combinations=list(gain_coverage_matrix.index)
    final_redundancy={}
    final_coverage_rate=base_coverage_rate
    for outer_iter in range(1,max_outer_iter+1):
        sensors_before=current_physical_sensors.copy(); print(f"\n******** 外层第 {outer_iter} 轮：增益系统删减 ********")
        active_sensors,selected_idx,selected_gain_matrix,removed_sensors,redundancy,coverage_rate=select_top_combinations_by_coverage(gain_coverage_matrix,current_physical_sensors,remaining_energy,required_num,consume,max_num=len(current_physical_sensors))
        current_physical_sensors=active_sensors
        each_iter_result[outer_iter]={"before_sensors":sensors_before.copy(),"remove":removed_sensors.copy(),"after_sensors":current_physical_sensors.copy(),"coverage":coverage_rate}
        final_gain_matrix=selected_gain_matrix
        final_combinations=selected_idx
        final_redundancy=redundancy
        final_coverage_rate=coverage_rate
        if not removed_sensors:
            print("\n重新设计增益系统后仍无传感器可删除，优化结束。")
            break
        print("本轮删除:",removed_sensors)
        print("本轮剩余:",current_physical_sensors)
        gain_coverage_matrix,new_coverage_rate,combination_map=design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold)
        new_covered_num=int(gain_coverage_matrix.any(axis=0).sum()) if not gain_coverage_matrix.empty else 0
        print("重新设计增益系统后的最大覆盖率:",round(new_coverage_rate,2),"%")
        if new_covered_num<required_num:
            print("警告：重构后的增益系统未达到初始固定覆盖约束，停止进一步删减。")
            break
    selected_combinations=[]
    for c in final_combinations:
        if isinstance(c,str): selected_combinations.append(list(ast.literal_eval(c)))
        else: selected_combinations.append(list(c))
    print("\n================ 最终结果 ================"); print("最终激活物理传感器:",current_physical_sensors); print("最终激活数量:",len(current_physical_sensors)); print("最终合作覆盖率:",round(final_coverage_rate,2),"%")
    return current_physical_sensors,selected_combinations,final_gain_matrix,final_redundancy,final_coverage_rate,each_iter_result




input_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_schedule_result.xlsx"
covariance_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx"
burst_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
gain_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_coverage_matrix.xlsx"

initial_energy=100
consume=1
sensitivity_threshold=-4.22068293050037


if __name__=="__main__":
    excel=pd.ExcelFile(input_file); sheet_names=excel.sheet_names
    df0=pd.read_excel(input_file,sheet_name=sheet_names[0],index_col=0)
    sensor_ids=list(df0.columns); n_sensor=len(sensor_ids); sensor_index={s:i for i,s in enumerate(sensor_ids)}
    all_sensor_energy={sensor:initial_energy for sensor in sensor_ids}
    gain_results=[]
    gain_coverage_results={}

    for sheet_idx,sheet in enumerate(sheet_names):
        available_sensors=[s for s in sensor_ids if all_sensor_energy[s]>=consume]
        covariance_matrix=pd.read_excel(covariance_file,sheet_name=sheet,index_col=0)
        pipe_burst_matrix=pd.read_excel(burst_file,sheet_name=sheet,index_col=0)
        temp_energy={s:all_sensor_energy[s] for s in available_sensors}

        final_active_ids,selected_combinations,gain_coverage_matrix,redundancy,coverage_rate,each_iter_result=iterative_remove_redundant_sensors(
            sensor_ids,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,temp_energy)

        gain_coverage_results[sheet]=gain_coverage_matrix

        final_x=np.zeros(n_sensor)
        for sensor in final_active_ids: final_x[sensor_index[sensor]]=1

        for sensor in final_active_ids:
            all_sensor_energy[sensor]-=consume
            all_sensor_energy[sensor]=max(all_sensor_energy[sensor],0)

        final_state={sensor:[int(final_x[sensor_index[sensor]]),round(all_sensor_energy[sensor],2)] for sensor in sensor_ids}
        final_active_energy=[all_sensor_energy[s] for s in final_active_ids]
        mean_energy=round(np.mean(final_active_energy),2) if final_active_energy else np.nan
        energy_range=round(max(final_active_energy)-min(final_active_energy),2) if final_active_energy else np.nan

        gain_results.append([sheet,final_state,",".join(map(str,final_active_ids)),len(final_active_ids),mean_energy,energy_range,
                             coverage_rate,selected_combinations,redundancy,each_iter_result])


        mode="w" if sheet_idx==0 else "a"
        with pd.ExcelWriter(gain_file,engine="openpyxl",mode=mode,**({} if mode=="w" else {"if_sheet_exists":"replace"})) as writer:
            gain_coverage_matrix.to_excel(writer,sheet_name=str(sheet)[:31])

        gain_df=pd.DataFrame([
            {"time":r[0],**{str(k):v for k,v in r[1].items()},"active_sensor":r[2],"active_number":r[3],
             "mean_energy":r[4],"energy_range":r[5],"coverage_rate":r[6],"selected_combinations":str(r[7]),
             "redundancy":str(r[8]),"iteration_result":str(r[9])}
            for r in gain_results
        ])

        with pd.ExcelWriter(output_file,engine="openpyxl",mode="w") as writer:
            gain_df.to_excel(writer,sheet_name="优化结果",index=False)

        min_energy=min(all_sensor_energy.values())

        print("时刻:",sheet,
              "激活传感器:",final_active_ids,
              "激活数量:",len(final_active_ids),
              "覆盖率:",coverage_rate,
              "最低剩余电量:",round(min_energy,2),
              "迭代过程:",each_iter_result)

        print(f"已保存 {sheet} 的优化结果")