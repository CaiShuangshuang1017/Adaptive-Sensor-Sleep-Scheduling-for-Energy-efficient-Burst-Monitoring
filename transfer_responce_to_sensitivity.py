#将爆管响应矩阵转化为灵敏度矩阵
import pandas as pd
import numpy as np

input_file = r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_H_decline.xlsx"
reference_file = r"D:\A_科研相关\小论文3_动态监测网络\L-town\exsiting_mean_std.xlsx"
output_file = r"D:\A_科研相关\小论文3_动态监测网络\L-town\all_time_sensitivity.xlsx"
threshold = -4.220682930500373


# 读取参考文件
ref_excel = pd.ExcelFile(reference_file)
ref_df = pd.read_excel(reference_file, sheet_name="std", header=None)

# 构建节点-参考值索引
# 第二列为节点编号
nodes = ref_df.iloc[2:, 1].astype(int).values

# 第2行，第3列开始为sheet编号1~24
sheet_names_ref = ref_df.iloc[1, 2:].astype(int).astype(str).values

# 保存参考值
ref_dict = {}
for i, node in enumerate(nodes):
    ref_dict[node] = {}
    for j, sheet in enumerate(sheet_names_ref):
        ref_dict[node][sheet] = ref_df.iloc[i+2, j+2]


# 读取原始excel
excel = pd.ExcelFile(input_file)

writer = pd.ExcelWriter(output_file, engine="openpyxl")

for sheet_name in map(str, range(1,25)):

    if sheet_name not in excel.sheet_names:
        print(f"跳过 {sheet_name}")
        continue

    df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)

    result = df.copy()

    # 第一行节点编号
    node_cols = df.iloc[0, 1:]

    for col_idx, node in enumerate(node_cols, start=1):

        if pd.isna(node):
            continue

        node = int(node)

        if node not in ref_dict:
            print(f"{sheet_name}: 节点{node}不存在参考值")
            continue

        divisor = ref_dict[node][sheet_name]

        if pd.isna(divisor) or divisor == 0:
            print(f"{sheet_name}: 节点{node}参考值异常")
            continue

        # 从第二行开始处理
        values = pd.to_numeric(
            df.iloc[1:, col_idx],
            errors="coerce"
        )

        normalized = values / float(divisor)

        binary = np.where(
            normalized < threshold,
            1,
            0
        )

        result.iloc[1:, col_idx] = binary


    # 保存sheet
    result.to_excel(
        writer,
        sheet_name=sheet_name,
        header=False,
        index=False
    )

    print(f"完成 {sheet_name}")

writer.close()

print("全部完成:", output_file)