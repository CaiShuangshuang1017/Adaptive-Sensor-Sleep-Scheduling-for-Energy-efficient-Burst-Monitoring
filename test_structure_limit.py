import numpy as np,pandas as pd,random,os
from multiprocessing import Pool

def initialize_population(population_size,current_physical_sensors,max_len=10):
    max_len=min(max_len,len(current_physical_sensors))
    return [random.sample(current_physical_sensors,random.randint(1,max_len)) for _ in range(population_size)]

def evaluate_population(population,covariance_matrix,burst_response):
    results=[]
    for individual in population:
        cov_sub=covariance_matrix.loc[individual,individual];std=np.sqrt(max(cov_sub.values.sum(),1e-10));decline=burst_response[individual].sum();results.append(decline/std)
    return results

def select(population,sensitivity_results):
    max_val=np.max(sensitivity_results);min_val=np.min(sensitivity_results);fitness=(max_val-sensitivity_results)/(max_val-min_val+1e-10)
    probabilities=fitness/fitness.sum() if fitness.sum()>0 else np.ones(len(population))/len(population)
    return [population[i] for i in np.random.choice(len(population),len(population),p=probabilities)]

def crossover(parent1,parent2,max_len=10):
    union=list(set(parent1)|set(parent2));random.shuffle(union);max_len=min(max_len,len(union))
    return union[:random.randint(1,max_len)],union[:random.randint(1,max_len)]

def mutate(individual,mutation_rate,current_physical_sensors,max_len=10):
    individual=individual.copy()
    if random.random()<mutation_rate:
        mutation_type=random.choice(["add","delete","swap"])
        if mutation_type=="add":
            available=list(set(current_physical_sensors)-set(individual))
            if available and len(individual)<max_len: individual.append(random.choice(available))
        elif mutation_type=="delete" and len(individual)>1: individual.remove(random.choice(individual))
        elif mutation_type=="swap":
            available=list(set(current_physical_sensors)-set(individual))
            if available: idx=random.randrange(len(individual));individual[idx]=random.choice(available)
    return individual

def genetic_algorithm(covariance_matrix,burst_response,population_size,current_physical_sensors,generations,mutation_rate,patience=10,tol=1e-6):
    population=initialize_population(population_size,current_physical_sensors);best=None;best_value=float("inf");no_improve=0
    for gen in range(generations):
        values=evaluate_population(population,covariance_matrix,burst_response);idx=np.argmin(values);current_best=values[idx]
        if current_best<best_value-tol:
            best_value=current_best;best=population[idx].copy();no_improve=0
        else:
            no_improve+=1
        if no_improve>=patience: break
        selected=select(population,values);new_population=[]
        for i in range(0,len(selected)-1,2):
            c1,c2=crossover(selected[i],selected[i+1]);new_population.extend([mutate(c1,mutation_rate,current_physical_sensors),mutate(c2,mutation_rate,current_physical_sensors)])
        population=new_population
    return best,best_value

def init_worker(pipe_array,sensors,cov_matrix,pop_size,generations,mutation_rate):
    global pipe_burst_array,sensor_pool,cov_matrix_global,population_size_global,generations_global,mutation_rate_global
    pipe_burst_array=pipe_array;sensor_pool=sensors;cov_matrix_global=cov_matrix;population_size_global=pop_size;generations_global=generations;mutation_rate_global=mutation_rate
    np.random.seed(os.getpid());random.seed(os.getpid())

def optimize_single_pipe(pipe_index):
    burst_response=pd.Series(pipe_burst_array[pipe_index],index=sensor_pool)
    best_combination,max_sensitivity=genetic_algorithm(cov_matrix_global,burst_response,population_size_global,sensor_pool,generations_global,mutation_rate_global)
    return pipe_index,best_combination,max_sensitivity

def select_high_energy_combinations(gain_coverage_matrix,combination_map,remaining_energy):
    if gain_coverage_matrix.empty: return gain_coverage_matrix.copy(),{},set()
    original_map=combination_map.copy();target_covered_num=int(gain_coverage_matrix.any(axis=0).sum())
    combination_energy={k:min(remaining_energy[s] for s in combination_map[k]) for k in gain_coverage_matrix.index}
    remaining=list(gain_coverage_matrix.index);selected=[];covered=pd.Series(False,index=gain_coverage_matrix.columns)
    while int(covered.sum())<target_covered_num and remaining:
        candidate_info=[]
        for k in remaining:
            row=gain_coverage_matrix.loc[k].astype(bool);new_cover_num=int((row&~covered).sum());candidate_info.append((k,combination_energy[k],new_cover_num))
        candidate_info.sort(key=lambda x:(-x[1],-x[2]));best,energy,new_cover_num=candidate_info[0]
        if new_cover_num==0: remaining.remove(best);continue
        selected.append(best);covered|=gain_coverage_matrix.loc[best].astype(bool);remaining.remove(best)
    new_G=gain_coverage_matrix.loc[selected].copy();new_map={k:combination_map[k] for k in selected}

    # 二次压缩：优先尝试删除低电量虚拟传感器，只要删除后覆盖仍满足要求就永久删除
    remove_order=sorted(list(new_G.index),key=lambda k:combination_energy[k])
    for k in remove_order:
        if k not in new_G.index: continue
        test_G=new_G.drop(index=k)
        test_covered_num=int(test_G.any(axis=0).sum()) if not test_G.empty else 0
        if test_covered_num>=target_covered_num:
            new_G=test_G;new_map.pop(k,None)
    before=set()
    for c in original_map.values(): before.update(c)
    after=set()
    for c in new_map.values(): after.update(c)
    removed_sensors=before-after
    return new_G,new_map,removed_sensors

def design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy):
    if not current_physical_sensors: return pd.DataFrame(columns=pipe_burst_matrix.index),0,{}
    pipe_array=pipe_burst_matrix.loc[:,current_physical_sensors].values;cov_matrix=covariance_matrix.loc[current_physical_sensors,current_physical_sensors]
    with Pool(processes=4,initializer=init_worker,initargs=(pipe_array,current_physical_sensors,cov_matrix,30,50,0.2)) as pool: results=pool.map(optimize_single_pipe,range(pipe_array.shape[0]))
    results.sort(key=lambda x:x[0]);all_pipe_best_combination=[r[1] for r in results if r[1]];unique_combinations=[];seen=set()
    for combination in all_pipe_best_combination:
        key=tuple(sorted(combination))
        if key not in seen: unique_combinations.append(list(key));seen.add(key)
    if not unique_combinations: return pd.DataFrame(columns=pipe_burst_matrix.index),0,{}
    sensor_index={s:i for i,s in enumerate(current_physical_sensors)};individual_info=[]
    for individual in unique_combinations:
        idx=np.array([sensor_index[i] for i in individual]);cov_sub=covariance_matrix.loc[individual,individual].values;std=np.sqrt(max(cov_sub.sum(),1e-10));individual_info.append((idx,std))
    gain_coverage_dict=[]
    for pipe_index in range(pipe_array.shape[0]):
        pipe_response=pipe_array[pipe_index];gain_sensitivity=[pipe_response[idx].sum()/std for idx,std in individual_info];gain_coverage_dict.append([1 if val<sensitivity_threshold else 0 for val in gain_sensitivity])
    combination_keys=[str(i) for i in unique_combinations];combination_map={str(i):i for i in unique_combinations}
    gain_coverage_matrix=pd.DataFrame(np.array(gain_coverage_dict).T,index=combination_keys,columns=pipe_burst_matrix.index);coverage_rate=gain_coverage_matrix.any(axis=0).mean()*100
    gain_coverage_matrix,combination_map,removed_sensors=select_high_energy_combinations(gain_coverage_matrix,combination_map,remaining_energy)
    coverage_rate=gain_coverage_matrix.any(axis=0).mean()*100
    return gain_coverage_matrix,coverage_rate,combination_map,removed_sensors

def check_structural_sensors_all_states(covariance_file,burst_file,original_coverage_file,sensitivity_threshold,output_file):
    excel=pd.ExcelFile(covariance_file);sheet_names=excel.sheet_names;all_results=[];state_summary=[]
    for state_idx,sheet in enumerate(sheet_names,1):
        print("\n"+"="*80);print(f"水力状态 {state_idx}/{len(sheet_names)}: {sheet}");print("="*80)
        covariance_matrix=pd.read_excel(covariance_file,sheet_name=sheet,index_col=0);pipe_burst_matrix=pd.read_excel(burst_file,sheet_name=sheet,index_col=0)
        sensor_ids=list(covariance_matrix.columns);remaining_energy={s:100 for s in sensor_ids}

        original_coverage=pd.read_excel(original_coverage_file,sheet_name=sheet,index_col=0)
        base_covered_num=int((original_coverage>0).any(axis=1).sum())
        base_rate=base_covered_num/original_coverage.shape[0]*100
        print("原系统参考覆盖管段数:",base_covered_num)
        print("原系统参考覆盖率:",round(base_rate,2),"%")

        structural_sensors=[];removable_sensors=[]
        for idx,sensor in enumerate(sensor_ids,1):
            print(f"[{idx}/{len(sensor_ids)}] 删除传感器 {sensor}")
            test_sensors=[s for s in sensor_ids if s!=sensor]
            test_G,test_rate,test_map,test_removed=design_gain_system(test_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy)
            test_covered_num=int(test_G.any(axis=0).sum()) if not test_G.empty else 0
            coverage_loss=max(0,base_covered_num-test_covered_num);structural_impact=coverage_loss/base_covered_num if base_covered_num>0 else 0
            is_structural=test_covered_num<base_covered_num
            if is_structural: structural_sensors.append(sensor);status="structural"
            else: removable_sensors.append(sensor);status="removable"
            all_results.append({"hydraulic_state":sheet,"state_index":state_idx,"sensor":sensor,"status":status,"structural":is_structural,"base_covered_num":base_covered_num,"test_covered_num":test_covered_num,"coverage_loss":coverage_loss,"structural_impact":structural_impact,"base_coverage_rate":base_rate,"test_coverage_rate":test_rate,"gsd_removed_after_delete":str(list(test_removed))})
            print(f"{sensor}: {status} | 覆盖 {test_covered_num}/{base_covered_num} | 损失 {coverage_loss}")

        state_summary.append({"hydraulic_state":sheet,"state_index":state_idx,"base_covered_num":base_covered_num,"base_coverage_rate":base_rate,"structural_number":len(structural_sensors),"removable_number":len(removable_sensors),"structural_sensors":str(structural_sensors),"removable_sensors":str(removable_sensors)})

    result_df=pd.DataFrame(all_results);summary_df=pd.DataFrame(state_summary)

    sensor_summary=result_df.groupby("sensor").agg(
        structural_times=("structural","sum"),
        mean_coverage_loss=("coverage_loss","mean"),
        max_coverage_loss=("coverage_loss","max"),
        mean_structural_impact=("structural_impact","mean"),
        max_structural_impact=("structural_impact","max")
    ).reset_index()

    sensor_summary["structural_ratio"]=sensor_summary["structural_times"]/len(sheet_names)
    sensor_summary["always_structural"]=sensor_summary["structural_times"]==len(sheet_names)
    sensor_summary["never_structural"]=sensor_summary["structural_times"]==0

    with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
        result_df.to_excel(writer,sheet_name="逐状态逐传感器",index=False)
        summary_df.to_excel(writer,sheet_name="各状态汇总",index=False)
        sensor_summary.to_excel(writer,sheet_name="传感器汇总",index=False)

    print("\n"+"="*80);print("24个水力状态检查完成");print("结果已保存:",output_file);print("="*80)
    return result_df,summary_df,sensor_summary


output_structural_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\structural_sensor_24states_ref_original.xlsx"
covariance_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx"
burst_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
original_coverage_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"

initial_energy=100;consume=1;sensitivity_threshold=-4.22068293050037;coverage_ratio=1;save_interval=1


if __name__=="__main__":
    result_df,summary_df,sensor_summary=check_structural_sensors_all_states(
        covariance_file,
        burst_file,
        original_coverage_file,
        sensitivity_threshold,
        output_structural_file
    )