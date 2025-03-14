import seaborn as sns
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块

# 示例数据
data = sns.load_dataset("tips")

# 绘制图表
sns.scatterplot(data=data, x="total_bill", y="tip")

# 显示图表
plt.show()