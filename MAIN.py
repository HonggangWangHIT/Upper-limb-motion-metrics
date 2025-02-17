import os
import numpy as np
import pandas as pd
from UP_Ldw import UP_Ldw

folder_path = os.path.join('.', 'Datasets', 'Arm-CODA')

# Randomly selected data #
RN_1 = np.random.permutation(16)[:10]
RN_2 = np.random.permutation(15)[:10]
result_FP_N = [f"{RN_1[i]}_{RN_2[i]}.csv" for i in range(len(RN_1))]
result_FP = [os.path.join(folder_path, result_FP_N[i]) for i in range(len(RN_1))]
pairs = [[result_FP[i], result_FP[i+1]] for i in range(0, len(result_FP), 2)]
Re_out = np.empty((len(pairs), 3), dtype=object)
j = 0

# Motion metrics #
for i in range(len(pairs)):
    Data1 = pd.read_csv(pairs[i][0]).values
    Data2 = pd.read_csv(pairs[i][1]).values
    in_data1 = Data1[:round(len(Data1)/3), [12, 13, 14, 48, 49, 50, 60, 61, 62]]
    in_data2 = Data2[:round(len(Data2)/3), [12, 13, 14, 48, 49, 50, 60, 61, 62]]
    Re_out[i, 0] = result_FP_N[j]
    Re_out[i, 1] = result_FP_N[j+1]
    Re_out[i, 2] = UP_Ldw(in_data1, in_data2)
    j += 2

print(Re_out)

# Save the result #
filename = 'Results.xlsx'
if not os.path.exists(filename):
    print(f"Creating new file: {filename}")
    pd.DataFrame(Re_out).to_excel(filename, index=False, header=False, engine='openpyxl')
else:
    try:
        existing_data = pd.read_excel(filename, header=None, engine='openpyxl').values
        new_data = np.vstack((existing_data, Re_out))
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            pd.DataFrame(new_data).to_excel(writer, index=False, header=False)
    except Exception as e:
        print(f"Error reading or writing file: {e}")
