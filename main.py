from sklearn import datasets,svm,metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from mlxtend.plotting import plot_decision_regions

#irisデータを読み込む
iris = datasets.load_iris()

#学習で使う特徴の選択（petal length, petal width）
X = iris.data[:, [2, 3]]
y = iris.target

#iris dataの可視化
sns.set(style="ticks")
df = sns.load_dataset("iris")
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'
sns.pairplot(df, hue='species', markers=["o", "s", "+"]).savefig('your_path/figure_2.png')

#トレニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1, stratify=y)

#SVMでトレーニングデータを学習させる
clf = svm.SVC()
clf.fit(X_train, y_train)

#テストデータを入れて分類させる
pre = clf.predict(X_test)
print(pre)

per = metrics.accuracy_score(y_test,pre)*100
print(str(per) + '[%]')

#グラフの描画
#NumPyからDataFrameに変換
s = pd.DataFrame(X_test)
w = s.rename(columns={0: 'petal length[cm]', 1: 'petal width[cm]'})
l = pd.DataFrame(pre)
q = l.rename(columns={0: 'species'})
p = pd.concat([w, q], axis=1)
print(p)


plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'
plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')


#sns.scatterplot(data = p,x='petal length[cm]', y='petal width[cm]',style = 'species')
plt.show()

