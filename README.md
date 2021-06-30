# Plot-in-reseach-paper

```python
%matplotlib notebook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#折线图
data = pd.read_csv('mnsit_accuracy.csv')
data = np.array(data)
data_u = pd.read_csv('mnist_accuracy_unlearning.csv')
data_u = np.array(data_u)
x = np.array(data[:,0])
y = np.array(data[:,1])
y_u = np.array(data_u[:,1])
plt.grid(linestyle = "--") 
plt.plot(x,y,'--',color = '#006BB2',linewidth=2.5, label="Retraining from Scratch")#s-:方形
plt.plot(x,y_u,'--',color = '#B22400',linewidth=2.5, label="Rapid Retraining")#s-:方形
#plt.xticks([0,1,2],['10','20','30'], fontsize=14)
plt.yticks( fontsize=14)
plt.xlabel("Retraining Round", fontsize=16)#横坐标名字
plt.ylabel("Accuracy (%)", fontsize=16)#纵坐标名字
plt.legend(loc = "best",fontsize=12)#图例
#plt.savefig('fig-8.pdf',dpi=600,bbox_inches = 'tight')
plt.show()
```
