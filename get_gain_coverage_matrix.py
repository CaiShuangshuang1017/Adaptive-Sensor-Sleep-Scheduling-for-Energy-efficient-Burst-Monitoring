import numpy as np,pandas as pd,random,os
from multiprocessing import Pool
from collections import Counter
import ast

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

def select_top_combinations_by_coverage(gain_coverage_matrix,max_num=20):
    coverage_count=gain_coverage_matrix.astype(bool).sum(axis=0)
    single_covered_num=(coverage_count==1).sum()
    print("只被覆盖一次的管段数量:",single_covered_num)
    selected=[]
    remaining=list(gain_coverage_matrix.index)
    covered=np.zeros(gain_coverage_matrix.shape[1],dtype=bool)
    while remaining and len(selected)<max_num:
        best=None
        best_gain=0
        for c in remaining:
            current=gain_coverage_matrix.loc[[c]].iloc[0].values.astype(bool)
            gain=np.sum(current & ~covered)
            if gain>best_gain:
                best_gain=gain
                best=c
        if best is None or best_gain==0:
            break
        selected.append(best)
        covered|=gain_coverage_matrix.loc[[best]].iloc[0].values.astype(bool)
        remaining.remove(best)
    return selected

def design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold):
    pipe_array=pipe_burst_matrix.loc[:,current_physical_sensors].values
    cov_matrix=covariance_matrix.loc[current_physical_sensors,current_physical_sensors]
    with Pool(processes=4,initializer=init_worker,initargs=(pipe_array,current_physical_sensors,cov_matrix,50,5,0.2)) as pool:
        results=pool.map(optimize_single_pipe,range(pipe_array.shape[0]))
    results.sort(key=lambda x:x[0])
    all_pipe_best_combination=[r[1] for r in results]

    unique_combinations=[]
    seen=set()
    for combination in all_pipe_best_combination:
        key=tuple(sorted(combination))
        if key not in seen:
            unique_combinations.append(list(key))
            seen.add(key)

    sensor_index={s:i for i,s in enumerate(current_physical_sensors)}
    individual_info=[]
    for individual in unique_combinations:
        idx=np.array([sensor_index[i] for i in individual])
        cov_sub=covariance_matrix.loc[individual,individual].values
        std=np.sqrt(max(cov_sub.sum(),1e-10))
        individual_info.append((idx,std))

    gain_coverage_dict=[]
    for pipe_index in range(pipe_array.shape[0]):
        pipe_response=pipe_array[pipe_index]
        gain_sensitivity=[pipe_response[idx].sum()/std for idx,std in individual_info]
        gain_coverage_dict.append([1 if val<sensitivity_threshold else 0 for val in gain_sensitivity])

    combination_keys=[str(i) for i in unique_combinations]
    combination_map={str(i):i for i in unique_combinations}

    gain_coverage_matrix_all=pd.DataFrame(
        np.array(gain_coverage_dict).T,
        index=combination_keys,
        columns=pipe_burst_matrix.index)

    selected_idx=select_top_combinations_by_coverage(gain_coverage_matrix_all,max_num=20)
    gain_coverage_matrix=gain_coverage_matrix_all.loc[selected_idx]
    selected_combinations=[combination_map[i] for i in selected_idx]

    coverage_rate=(gain_coverage_matrix.sum(axis=0)>0).mean()*100
    return selected_combinations,gain_coverage_matrix,coverage_rate

def calculate_sensor_redundancy(selected_combinations,gain_coverage_matrix):
    gain_sensors=sorted(list(set([s for c in selected_combinations for s in c])))
    redundancy={}
    coverage_array=gain_coverage_matrix.values.astype(int)
    for sensor in gain_sensors:
        contain_idx=[i for i,c in enumerate(selected_combinations) if sensor in c]
        noncontain_idx=[i for i,c in enumerate(selected_combinations) if sensor not in c]
        contain=np.any(coverage_array[contain_idx,:]==1,axis=0)
        other=np.any(coverage_array[noncontain_idx,:]==1,axis=0) if noncontain_idx else np.zeros(coverage_array.shape[1],dtype=bool)
        redundancy[sensor]=np.logical_and(contain,other).sum()/max(contain.sum(),1)
    return redundancy

def iterative_remove_redundant_sensors(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy):
    current_physical_sensors=current_physical_sensors.copy()
    each_iter_result={}
    iter=1
    while True:
        selected_combinations,gain_coverage_matrix,coverage_rate=design_gain_system(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold)
        gain_sensors=sorted(list(set([s for c in selected_combinations for s in c])))
        not_used_sensors=[s for s in current_physical_sensors if s not in gain_sensors]
        for sensor in not_used_sensors:
            remaining_energy.pop(sensor,None)
        current_physical_sensors=gain_sensors
        redundancy=calculate_sensor_redundancy(selected_combinations,gain_coverage_matrix)
        redundant=[s for s,v in redundancy.items() if v>=1-1e-10]
        current_iter_remove_sensor=not_used_sensors.copy()
        if redundant:
            min_energy=min(remaining_energy[s] for s in redundant)
            candidate=[s for s in redundant if remaining_energy[s]==min_energy]
            remove_sensor=random.choice(candidate)
            current_physical_sensors.remove(remove_sensor)
            remaining_energy.pop(remove_sensor)
            current_iter_remove_sensor.append(remove_sensor)
        each_iter_result[iter]={'coverage':coverage_rate,'remove':current_iter_remove_sensor}
        iter+=1
        if not current_iter_remove_sensor:
            break
    return current_physical_sensors,selected_combinations,gain_coverage_matrix,redundancy,coverage_rate,each_iter_result

# if __name__=="__main__":
#     #freeze_support()
#     current_time='1'
#     current_physical_sensors=[1,4,54,114,163,229,288,296,332,342,410,415,458,469,495,506,516,519,549,613,644,679,722,726,740,752,769]
#     sensitivity_threshold=-4.22068293050037
#     pipe_burst_matrix=pd.read_excel(r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx",sheet_name=current_time,index_col=0)
#     covariance_matrix=pd.read_excel(r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx",sheet_name=current_time,index_col=0)
#     remaining_energy={s:99.0 for s in current_physical_sensors}
#     current_physical_sensors,selected_combinations,gain_coverage_matrix,redundancy,coverage_rate,each_iter_result=iterative_remove_redundant_sensors(current_physical_sensors,covariance_matrix,pipe_burst_matrix,sensitivity_threshold,remaining_energy)