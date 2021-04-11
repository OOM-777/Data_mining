import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

data = pd.read_csv('科技区2019-1_2020-3.csv')

# print(data)
sci = data.loc[data['分区'] == '科学科普']
sci = sci[['分区', 'author', 'coins', 'danmu', 'favorite', 'likes', 'replay', 'share', 'view', 'title', 'date']]

# I：interaction_rate 反应的是平均每个视频的互动率
# F：Frequence 表示的是每个视频的平均发布周期
# L：like_rate 表示的是统计时间内发布视频的平均点赞率

count = sci.groupby("author")['分区'].count().reset_index()
count.columns = ['author', 'times']
com_m = pd.merge(count, sci, on='author', how='inner')
com = com_m[com_m['times'] >= 5]

com['date'] = pd.to_datetime(com.date)
last = com.groupby("author")['date'].max()
late = com.groupby("author")['date'].min()
F = round((last - late).dt.days / com.groupby("author")['date'].count()).reset_index()
F.columns = ['author', 'F']
F = pd.merge(com, F, on='author', how='inner')

F.loc[F['F'].idxmin()]
F = F.loc[F['F'] > 0]

danmu = F.groupby('author')['danmu'].sum()
reply = F.groupby('author')['replay'].sum()
view = F.groupby("author")['view'].sum()
count = F.groupby("author")['date'].count()
I = round((danmu + reply) / view / count * 100, 2).reset_index()
I.columns = ['author', 'I']
I = pd.merge(F, I, on='author', how='inner')

I['xixi'] = (I['likes'] + I['coins'] * 2 + I['favorite'] * 3 + I['share']) / I['view'] * 100
L = (I.groupby("author")['xixi'].sum() / I.groupby("author")['date'].count()).reset_index()
L.columns = ['author', 'L']
IFL = pd.merge(I, L, on='author', how='inner')
IFL = IFL[['author', 'I', 'F', 'L']]

IFL['I_score'] = pd.cut(IFL['I'], bins=[0, 0.03, 0.06, 0.11, 1000], labels=[1, 2, 3, 4], right=False).astype(float)
IFL['F_score'] = pd.cut(IFL['F'], bins=[0, 7, 15, 30, 90, 1000], labels=[5, 4, 3, 2, 1], right=False).astype(float)
IFL['L_score'] = pd.cut(IFL['L'], bins=[0, 5.39, 9.07, 15.58, 1000], labels=[1, 2, 3, 4], right=False).astype(float)
print(IFL)
IFL = IFL.drop_duplicates()

print("===============================")
print(IFL)

sacle_matrix = IFL.iloc[:, 1:4]
print(sacle_matrix)

model_scaler = MinMaxScaler()
data_scaled = model_scaler.fit_transform(sacle_matrix)
print(data_scaled.round(2))

# 用循环多次聚类，获取最优的K值（轮廓系数决定）
score_list = list()  # 定义空列表保存每次K值及其平均轮廓系数记录
silhouette_int = -1
for n_clusters in range(2, 10):
    model_kmeans = KMeans(n_clusters=n_clusters)  # 建立聚类模型对象
    labels_tmp = model_kmeans.fit_predict(data_scaled)  # 训练聚类模型
    silhouette_tmp = silhouette_score(data_scaled, labels_tmp)  # 得到每个K下的平均轮廓系数
    if silhouette_tmp > silhouette_int:  # 如果平均轮廓系数更高
        best_k = n_clusters  # 保存最佳的K值
        silhouette_int = silhouette_tmp
        best_kmeans = model_kmeans  # 保存模型实例对象
        cluster_labels_k = labels_tmp  # 保存聚类标签
    score_list.append([n_clusters, silhouette_tmp])  # 将每次K值及其平均轮廓系数记录

print(score_list)

print('最优的K值是：{0} \n对应的轮廓系数是{1}'.format(best_k, silhouette_int))

cluster_labels = pd.DataFrame(cluster_labels_k, columns=['clusters'])
print(cluster_labels)

# 连接IFL数据集和簇类标签，生成聚类结果
merge_data = pd.concat((IFL.iloc[:, 1:4], cluster_labels), axis=1)
print(merge_data)
exit()
cluster_count = merge_data.groupby('clusters')['author'].count()
print(cluster_count)

# 计算每个类别下的样本占比
cluster_ratio = (cluster_count / len(merge_data)).round(2)
cluster_ratio.name = 'percentage'
print(cluster_ratio)

# 计算各个聚类类别内部最显著的特征值
cluster_features = []  # 记录每个类别下的I F L均值
for line in range(best_k):
    label_data = merge_data[merge_data['clusters'] == line].iloc[:, 1:4]  # 按簇类标签选取IFL数据
    desc_data = label_data.describe().round(3)  # 获取描述统计
    mean_data = desc_data.iloc[2, :]  # 获取均值
    mean_data.name = line  # 每行标签名称
    cluster_features.append(mean_data)

cluster_pd = pd.DataFrame(cluster_features)
print(cluster_pd)
