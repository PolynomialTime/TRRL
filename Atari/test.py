import pandas as pd
import os

# 定义文件夹路径
input_folder = 'E:\TRRL\csv\PongNoFrameskip-v4' # 输入文件夹路径
output_folder = 'E:\TRRL\csv\PongNoFrameskip-v4\ reward'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 读取文件
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        
        # 只保留前300行
        df = df.head(65)
        
        # 保存到输出文件夹
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)

print("处理完成！")