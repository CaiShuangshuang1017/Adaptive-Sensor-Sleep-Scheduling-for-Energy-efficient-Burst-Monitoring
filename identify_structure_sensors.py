import numpy as np,pandas as pd,random,os
from multiprocessing import Pool
from collections import Counter
import ast
from multiprocessing import freeze_support

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
    with Pool(processes=4,initializer=init_worker,initargs=(pipe_array,current_physical_sensors,cov_matrix,10,5,0.2)) as pool:
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

    selected_idx=selected_idx=select_top_combinations_by_coverage(gain_coverage_matrix_all,max_num=20)
    gain_coverage_matrix=gain_coverage_matrix_all.loc[selected_idx]
    selected_combinations=[combination_map[i] for i in selected_idx]
    coverage_rate=(gain_coverage_matrix.sum(axis=0)>0).mean()*100
    return selected_combinations,gain_coverage_matrix,coverage_rate

def identify_structural_always_on_sensors(current_physical_sensors,
                                          covariance_matrix,
                                          pipe_burst_matrix,
                                          sensitivity_threshold,
                                          coverage_ratio=0.95):
    """
    识别当前水力状态下的结构性常开传感器。

    判定原则：
    1. 先用全部当前物理传感器构建合作增益系统，得到基准合作覆盖率；
    2. 逐个移除物理传感器 s；
    3. 用剩余物理传感器重新构建合作增益系统；
    4. 若移除 s 后最大可实现覆盖率低于基准覆盖率的 coverage_ratio，
       则认为 s 在当前水力状态下不可被其他物理传感器及合作增益替代，
       判定为结构性常开传感器。
    """

    current_physical_sensors=current_physical_sensors.copy()

    # ==========================================================
    # 1. 全部传感器条件下的基准合作监测性能
    # ==========================================================


    base_combinations,base_gain_matrix,base_coverage_rate=design_gain_system(
        current_physical_sensors,
        covariance_matrix,
        pipe_burst_matrix,
        sensitivity_threshold)

    required_coverage=base_coverage_rate*coverage_ratio

    print("\n========== 结构性常开传感器识别 ==========")
    print("基准合作覆盖率:",round(base_coverage_rate,2),"%")
    print("最低允许覆盖率:",round(required_coverage,2),"%")

    structural_sensors=[]
    replaceable_sensors=[]
    result={}

    # ==========================================================
    # 2. Leave-one-out：逐个移除传感器并重构增益系统
    # ==========================================================
    for sensor in current_physical_sensors:

        test_sensors=[
            s for s in current_physical_sensors
            if s!=sensor
        ]


        # 如果已经没有传感器
        if not test_sensors:
            structural_sensors.append(sensor)
            result[sensor]={
                "coverage_without_sensor":0,
                "coverage_loss":base_coverage_rate,
                "replaceable":False
            }
            continue

        try:
            _,_,test_coverage_rate=design_gain_system(
                test_sensors,
                covariance_matrix,
                pipe_burst_matrix,
                sensitivity_threshold)
            
        except Exception as e:
            # 删除后无法构建设计增益系统，也视为不可替代
            print(f"传感器 {sensor}: 删除后增益系统构建失败 -> {e}")

            structural_sensors.append(sensor)

            result[sensor]={
                "coverage_without_sensor":0,
                "coverage_loss":base_coverage_rate,
                "replaceable":False
            }
            continue

        coverage_loss=base_coverage_rate-test_coverage_rate

        # ======================================================
        # 3. 判断是否满足性能保持要求
        # ======================================================
        if test_coverage_rate<required_coverage:
            structural_sensors.append(sensor)
            replaceable=False
        else:
            replaceable_sensors.append(sensor)
            replaceable=True

        result[sensor]={
            "coverage_without_sensor":test_coverage_rate,
            "coverage_loss":coverage_loss,
            "replaceable":replaceable
        }

        print(
            f"传感器 {sensor}: "
            f"删除后覆盖率={test_coverage_rate:.2f}% | "
            f"下降={coverage_loss:.2f}% | "
            f"{'结构性常开' if not replaceable else '可替代'}"
        )

    print("\n结构性常开传感器:",
          structural_sensors)

    print("可被替代传感器:",
          replaceable_sensors)

    return structural_sensors,replaceable_sensors,result

input_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_coverage.xlsx"
output_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_schedule_result.xlsx"
covariance_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_covariance.xlsx"
burst_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
gain_file=r"D:\A_科研相关\小论文3_动态监测网络\L-town\optimized_gain_coverage_matrix.xlsx"

initial_energy=100
consume=1
sensitivity_threshold=-4.22068293050037
excel=pd.ExcelFile(input_file)
sheet_names=excel.sheet_names
df0=pd.read_excel(input_file,sheet_name=sheet_names[0],index_col=0)

sensor_ids=list(df0.columns)
n_sensor=len(sensor_ids)
sensor_index={s:i for i,s in enumerate(sensor_ids)}



if __name__=="__main__":
    freeze_support()
    for sheet_idx,sheet in enumerate(sheet_names):
        covariance_matrix=pd.read_excel(covariance_file,sheet_name=sheet,index_col=0)
        pipe_burst_matrix=pd.read_excel(burst_file,sheet_name=sheet,index_col=0)

        structural_sensors,replaceable_sensors,structural_result=\
            identify_structural_always_on_sensors(
                sensor_ids,
                covariance_matrix,
                pipe_burst_matrix,
                sensitivity_threshold,
                coverage_ratio=0.95)