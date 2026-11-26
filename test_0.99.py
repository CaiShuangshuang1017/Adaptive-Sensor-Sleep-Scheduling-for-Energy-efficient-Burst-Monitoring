import numpy as np,pandas as pd,random,os,ast
from multiprocessing import Pool

pipe_burst_array=None;sensor_pool=None;cov_matrix_global=None;population_size_global=None;generations_global=None;mutation_rate_global=None

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
    with Pool(processes=4,initializer=init_worker,initargs=(pipe_array,current_physical_sensors,cov_matrix,50,50,0.2)) as pool: results=pool.map(optimize_single_pipe,range(pipe_array.shape[0]))
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
    return gain_coverage_matrix,coverage_rate,combination_map

def calculate_sensor_redundancy(gain_coverage_matrix,current_physical_sensors):
    G=gain_coverage_matrix.astype(bool).copy();combinations=list(G.index);combination_sensor_dict={}
    for c in combinations:
        if isinstance(c,(tuple,list,set)): combination_sensor_dict[c]=list(c)
        elif isinstance(c,str): combination_sensor_dict[c]=list(ast.literal_eval(c))
        else: raise ValueError(f"无法解析组合:{c}")
    redundancy={}
    for sensor in current_physical_sensors:
        contain=[c for c in combinations if sensor in combination_sensor_dict[c]];noncontain=[c for c in combinations if sensor not in combination_sensor_dict[c]]
        if not contain: redundancy[sensor]=1.0;continue
        contain_cover=G.loc[contain].any(axis=0);other_cover=G.loc[noncontain].any(axis=0) if noncontain else pd.Series(False,index=G.columns);contain_num=int(contain_cover.sum())
        redundancy[sensor]=float((contain_cover&other_cover).sum()/contain_num) if contain_num>0 else 1.0
    return redundancy

def remove_sensor_combinations(sensor,gain_coverage_matrix,combination_map):
    remove_rows=[c for c in gain_coverage_matrix.index if sensor in (combination_map[c] if c in combination_map else ast.literal_eval(c))]
    new_G=gain_coverage_matrix.drop(index=remove_rows);new_map={c:combination_map[c] for c in new_G.index if c in combination_map}
    return new_G,new_map

def iterative_remove_redundant_sensors(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy,current_original_coverage):
    current_physical_sensors=current_physical_sensors.copy();each_iter_result={};failed_sensors=set()
    gain_coverage_matrix,base_coverage_rate,combination_map=design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy)
    n_pipe=gain_coverage_matrix.shape[1];base_covered_num=int(gain_coverage_matrix.any(axis=0).sum()) if not gain_coverage_matrix.empty else 0;required_num=int(max(np.ceil(base_covered_num*coverage_ratio),np.ceil(n_pipe*0.9),np.ceil(n_pipe*current_original_coverage)));required_rate=required_num/n_pipe*100 if n_pipe>0 else 0
    print("全部传感器参考覆盖管段数:",base_covered_num);print("全部传感器参考覆盖率:",round(base_coverage_rate,2),"%");print("性能保持比例:",coverage_ratio);print("最低要求覆盖管段数:",required_num);print("最低要求覆盖率:",round(required_rate,2),"%")
    outer_iter=0
    while True:
        outer_iter+=1;redundancy=calculate_sensor_redundancy(gain_coverage_matrix,current_physical_sensors);available_candidates=[s for s in current_physical_sensors if s not in failed_sensors]
        if not available_candidates: print("所有剩余传感器均已验证为不可删除，物理冗余消除结束。");break
        random_order={s:random.random() for s in available_candidates};candidates=sorted(available_candidates,key=lambda s:(remaining_energy[s],-redundancy.get(s,0),random_order[s]))
        print(f"\n******** 第 {outer_iter} 轮 ********");print("已确认不可删除:",sorted(failed_sensors));print("候选顺序:",[(s,round(remaining_energy[s],2),round(redundancy.get(s,0),3)) for s in candidates]);removed_this_round=False
        for sensor in candidates:
            print(f"\n测试:{sensor} | 电量:{remaining_energy[sensor]:.2f} | 冗余度:{redundancy.get(sensor,0):.3f}")
            test_G,test_map=remove_sensor_combinations(sensor,gain_coverage_matrix,combination_map);current_covered_num=int(test_G.any(axis=0).sum()) if not test_G.empty else 0;current_coverage_rate=current_covered_num/n_pipe*100 if n_pipe>0 else 0
            if current_covered_num>=required_num:
                sensors_before=current_physical_sensors.copy();current_physical_sensors=[s for s in current_physical_sensors if s!=sensor];gain_coverage_matrix=test_G;combination_map=test_map
                each_iter_result[outer_iter]={"remove":[sensor],"remove_type":"fast_remove","remove_energy":remaining_energy[sensor],"remove_redundancy":redundancy.get(sensor,0),"before_sensors":sensors_before,"after_sensors":current_physical_sensors.copy(),"coverage":current_coverage_rate,"covered_num":current_covered_num}
                print(f"快速删除成功:{sensor} | 覆盖率:{current_coverage_rate:.2f}%");removed_this_round=True;break
            test_sensors=[s for s in current_physical_sensors if s!=sensor];new_G,new_coverage_rate,new_map=design_gain_system(test_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy);new_covered_num=int(new_G.any(axis=0).sum()) if not new_G.empty else 0
            if new_covered_num<required_num:
                failed_sensors.add(sensor);print(f"{sensor} 重构仍不达标 → 标记结构限制，后续不再测试");continue
            sensors_before=current_physical_sensors.copy();current_physical_sensors=test_sensors;gain_coverage_matrix=new_G;combination_map=new_map
            each_iter_result[outer_iter]={"remove":[sensor],"remove_type":"redesign_remove","remove_energy":remaining_energy[sensor],"remove_redundancy":redundancy.get(sensor,0),"before_sensors":sensors_before,"after_sensors":current_physical_sensors.copy(),"coverage":new_coverage_rate,"covered_num":new_covered_num}
            print(f"重构删除成功:{sensor} | 覆盖率:{new_coverage_rate:.2f}%");removed_this_round=True;break
        if not removed_this_round: print("所有当前候选均无法继续删除，物理冗余消除结束。");break

    # 迭代结束后，基于最终剩余物理传感器重新构建完整GS
    final_full_G,final_full_rate,final_full_map=design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy)
    final_full_num=int(final_full_G.any(axis=0).sum()) if not final_full_G.empty else 0

    # 若随机GSD本次重构结果异常低于性能要求，则保留上一轮有效GS
    if final_full_num<required_num:
        print("最终重新GSD结果低于性能约束，采用迭代结束时已有GS进行压缩")
        final_full_G=gain_coverage_matrix.copy();final_full_map=combination_map.copy();final_full_rate=final_full_G.any(axis=0).mean()*100 if not final_full_G.empty else 0

    # 最后一步才进行GS压缩
    compressed_G,compressed_map,_=select_high_energy_combinations(final_full_G,final_full_map,remaining_energy)

    # 压缩GS实际使用的物理传感器
    used_sensors=set()
    for combination in compressed_map.values(): used_sensors.update(combination)

    # 最终GS未使用的物理传感器可进一步休眠
    gs_sleep_sensors=set(current_physical_sensors)-used_sensors
    final_sensors=[s for s in current_physical_sensors if s in used_sensors]
    final_gain_matrix=compressed_G.copy();final_coverage_rate=final_gain_matrix.any(axis=0).mean()*100 if not final_gain_matrix.empty else 0
    selected_combinations=[list(v) for v in compressed_map.values()]

    each_iter_result["final_GS_compression"]={"before_sensors":current_physical_sensors.copy(),"GS_sleep_sensors":list(gs_sleep_sensors),"after_sensors":final_sensors.copy(),"GS_before_number":len(final_full_G),"GS_after_number":len(compressed_G),"coverage_before":final_full_rate,"coverage_after":final_coverage_rate}
    print("\n******** 最终GS压缩 ********");print("物理冗余消除后传感器:",current_physical_sensors);print("压缩前虚拟传感器数量:",len(final_full_G));print("压缩后虚拟传感器数量:",len(compressed_G));print("GS压缩进一步休眠传感器:",sorted(gs_sleep_sensors));print("最终激活传感器:",final_sensors);print("最终覆盖率:",round(final_coverage_rate,2),"%")

    final_redundancy=calculate_sensor_redundancy(final_gain_matrix,final_sensors)
    return final_sensors,selected_combinations,final_gain_matrix,final_redundancy,final_coverage_rate,each_iter_result,base_coverage_rate,base_covered_num,required_num

def save_results(gain_results,gain_coverage_results,output_file,gain_file):
    if gain_results:
        pd.DataFrame(gain_results).to_excel(output_file,sheet_name="优化结果",index=False)
    if gain_coverage_results:
        with pd.ExcelWriter(gain_file,engine="openpyxl",mode="w") as writer:
            for key,G in gain_coverage_results.items(): G.to_excel(writer,sheet_name=key[:31])
    print(f">>> 已保存累计 {len(gain_results)} 个调度周期结果")

input_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\improved_gain_schedule_result_u_0.95.xlsx"
covariance_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx"
burst_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
gain_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\improved_gain_coverage_matrix_u_0.95.xlsx"

initial_energy=100;consume=1;sensitivity_threshold=-4.22068293050037;coverage_ratio=0.95;save_interval=1

if __name__=="__main__":
    excel=pd.ExcelFile(input_file);sheet_names=excel.sheet_names;df0=pd.read_excel(input_file,sheet_name=sheet_names[0],index_col=0);sensor_ids=list(df0.columns);n_sensor=len(sensor_ids);sensor_index={s:i for i,s in enumerate(sensor_ids)}
    all_sensor_energy={sensor:initial_energy for sensor in sensor_ids};gain_results=[];gain_coverage_results={};system_failure_time=None;depleted_final=[];global_step=0;cycle=0;stop_system=False
    original_coverage=[0.850828729,0.790055249,0.654143646,0.639779006,0.679558011,0.745856354,0.895027624,0.922651934,0.933701657,0.948066298,0.939226519,0.944751381,0.934806630,0.921546961,0.913812155,0.908287293,0.876243094,0.948066298,0.954696133,0.950276243,0.954696133,0.953591160,0.943646409,0.928176796]
    try:
        while not stop_system:
            cycle+=1;print("\n"+"="*80);print(f"开始第 {cycle} 个24状态循环");print("="*80)
            for state_idx,sheet in enumerate(sheet_names,1):
                current_original_coverage=original_coverage[state_idx-1]
                global_step+=1;print("\n"+"#"*70);print(f"全局时刻:{global_step} | 周期:{cycle} | 状态:{state_idx}/{len(sheet_names)} | sheet:{sheet}");print("#"*70)
                covariance_matrix=pd.read_excel(covariance_file,sheet_name=sheet,index_col=0);pipe_burst_matrix=pd.read_excel(burst_file,sheet_name=sheet,index_col=0)
                available_sensors=sensor_ids.copy();temp_energy={s:all_sensor_energy[s] for s in available_sensors}
                result=iterative_remove_redundant_sensors(available_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,temp_energy,current_original_coverage)
                final_active_ids,selected_combinations,gain_coverage_matrix,redundancy,coverage_rate,each_iter_result,ref_rate,ref_covered_num,required_num=result
                final_x=np.zeros(n_sensor,dtype=int)
                for sensor in final_active_ids: final_x[sensor_index[sensor]]=1
                for sensor in final_active_ids: all_sensor_energy[sensor]=max(all_sensor_energy[sensor]-consume,0)
                depleted_sensors=[s for s in sensor_ids if all_sensor_energy[s]<=0];energy_failure=len(depleted_sensors)>0
                if energy_failure:
                    system_failure_time=global_step;depleted_final=depleted_sensors.copy()
                final_state={sensor:[int(final_x[sensor_index[sensor]]),round(all_sensor_energy[sensor],2)] for sensor in sensor_ids};final_active_energy=[all_sensor_energy[s] for s in final_active_ids];mean_energy=round(np.mean(final_active_energy),2) if final_active_energy else np.nan
                all_energy_values=list(all_sensor_energy.values());energy_range=round(max(all_energy_values)-min(all_energy_values),2);energy_std=round(float(np.std(all_energy_values)),3);min_energy=round(min(all_energy_values),2);max_energy=round(max(all_energy_values),2)
                gain_key=f"t{global_step}_c{cycle}_s{state_idx}";gain_coverage_results[gain_key]=gain_coverage_matrix
                gain_results.append({"step":global_step,"cycle":cycle,"state_index":state_idx,"hydraulic_state":sheet,"active_sensor":",".join(map(str,final_active_ids)),"active_number":len(final_active_ids),"active_mean_energy":mean_energy,"energy_range":energy_range,"energy_std":energy_std,"min_energy":min_energy,"max_energy":max_energy,"coverage_rate":coverage_rate,"reference_coverage_rate":ref_rate,"reference_covered_num":ref_covered_num,"required_covered_num":required_num,"energy_failure":energy_failure,"depleted_sensors":",".join(map(str,depleted_sensors)),"selected_combinations":str(selected_combinations),"redundancy":str(redundancy),"iteration_result":str(each_iter_result),"sensor_state":str(final_state)})
                print("\n---------------- 当前时刻最终结果 ----------------");print("全局时刻:",global_step);print("周期:",cycle);print("水力状态:",sheet);print("全部传感器参考覆盖率:",round(ref_rate,2),"%");print("固定最低覆盖管段数:",required_num);print("激活传感器:",final_active_ids);print("激活数量:",len(final_active_ids));print("优化后合作覆盖率:",round(coverage_rate,2),"%");print("最低剩余电量:",min_energy);print("最高剩余电量:",max_energy);print("全网电量极差:",energy_range);print("全网电量标准差:",energy_std)
                if len(gain_results)%save_interval==0: save_results(gain_results,gain_coverage_results,output_file,gain_file)
                if energy_failure:
                    print("\n"+"!"*70);print(f"监测寿命终止：第 {global_step} 个调度周期首次出现传感器电量归0");print("电量耗尽传感器:",depleted_sensors);print("!"*70)
                    save_results(gain_results,gain_coverage_results,output_file,gain_file);stop_system=True;break
    except Exception as e:
        print("\n程序异常:",e);print("正在保存已完成结果...")
        save_results(gain_results,gain_coverage_results,output_file,gain_file)
        raise
    if gain_results: save_results(gain_results,gain_coverage_results,output_file,gain_file)
    print("\n"+"="*80)
    if system_failure_time is None: print("模拟结束，但未出现传感器电量归0。")
    else:
        print("监测寿命:",system_failure_time,"个调度周期");print("寿命终止循环:",cycle);print("首批电量耗尽传感器:",depleted_final)
    print("最终剩余电量:",all_sensor_energy);print("="*80)