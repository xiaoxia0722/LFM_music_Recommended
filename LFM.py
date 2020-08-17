# !/usr/bin/python3
# @Author: XiaoXia
# @Time    : 2020/8/14 10:13
# @File    : LFM.py
# @Site    : 
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def LFM(R: np.ndarray, P: np.ndarray, Q: np.ndarray, K: int, steps=5000, learning_rate: float = 0.01, beta: float = 0.02, min_loss: float = 0.001, min_interval: float = 1e-3):
    """
    LFM算法
    :param min_interval: 梯度下降时,每次下降的最小损失差, 小于则停止梯度下降
    :param R: 用户-歌曲矩阵
    :param P: 用户隐语义矩阵
    :param Q: 歌曲隐语义矩阵
    :param K: 隐类数量
    :param steps: 学习迭代次数
    :param learning_rate: 学习区
    :param beta: 正则化项系数
    :param min_loss:  最小损失值
    :return: 学习后的P和Q
    """
    Q = Q.T
    loss = [get_loss(R, P, Q, K)]
    for t in range(steps):
        for i in range(len(R)):
            for j in range(len(R[0])):
                eij = R[i][j] - P[i, :] @ Q[:, j]
                for k in range(K):
                    P[i][k] = P[i][k] + learning_rate * (eij * Q[k][j] - beta * P[i][k])
                    Q[k][j] = Q[k][j] + learning_rate * (eij * P[i][k] - beta * Q[k][j])
        loss.append(get_loss(R, P, Q, K, beta))
        print("第%d次迭代的损失值为:%lf" % (t+1, loss[-1]))
        if loss[-1] <= min_loss:
            break
        if abs(loss[-1] - loss[-2]) <= min_interval:
            break
    return P, Q.T, loss


def get_loss(R: np.array, P: np.array, Q: np.array, K: int, beta=0.02):
    """
    求损失值
    :param R: 用户-歌曲矩阵
    :param P: 用户隐语义矩阵
    :param Q: 歌曲隐语义矩阵
    :param K: 隐类数量
    :param beta: 正则化项系数
    :return:
    """
    R_new = P @ Q
    loss = beta / 2 * (P ** 2).sum() + beta / 2 * (Q ** 2).sum()
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] != 0:
                loss += (R[i][j] - R_new[i][j]) ** 2
    return loss


def read_data(path: str):
    """
    读取csv文件
    :param path: 文件路径
    :return: 读取到的csv文件的DataFrame
    """
    return pd.read_csv(path)


def get_R(data: pd.DataFrame):
    """
    对数据进行处理并转换为用户-音乐矩阵R
    :param data:
    :return: 用户-音乐矩阵R(DataFrame)
    """
    # 以歌曲播放量占用户总播放量的比例为评分标准
    all_count = data.groupby('user')['play_count'].sum().reset_index()
    all_count.columns = ['user', 'all_count']
    new_data = pd.merge(data, all_count, on='user')
    new_data.loc[:, 'score'] = new_data.loc[:, 'play_count'] / new_data.loc[:, 'all_count']

    # 以播放最多的歌曲为1分,其余的歌曲的分数为播放次数占播放最多的歌曲的次数的比例
    # max_count_data = data.groupby('user')['play_count'].max().reset_index()
    # max_count_data.columns = ['user', 'max_count']
    # new_data = data.merge(max_count_data, on='user')
    # new_data.loc[:, 'score'] = new_data.loc[:, 'play_count'] / new_data.loc[:, 'max_count']

    R = new_data.pivot_table(index='user', columns='song', values='score')
    R = R.fillna(0)
    return R


def init_P_Q(R: np.array, K: int):
    """
    初始化LFM的两个矩阵P和Q
    :param R:
    :param K:
    :return:
    """
    P = np.random.rand(len(R), K)
    Q = np.random.rand(len(R[0]), K)
    return P, Q


def plot_loss(loss, learning_rate=0.01):
    """
    绘制损失函数曲线
    :param loss:
    :param learning_rate:
    :return:
    """
    plt.plot([i * learning_rate for i in range(len(loss))], loss)
    plt.title('loss learning curve')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


def recommended_song(R, R_hat, user):
    """
    根据推荐结果进行推荐
    :param R: 原用户-歌曲矩阵
    :param R_hat: 预测后的用户歌曲矩阵
    :param user: 需要推荐的用户
    :return:
    """
    recommended = R_hat.loc[user, :]
    recommended.sort_values(ascending=False, inplace=True)
    recommended_list = recommended.index.to_list()
    recommended_score = list(recommended.values)
    songs = pd.read_csv('./data/song.csv', index_col=0)
    print('推荐评分前50个(包括以前听过的):')
    for i in range(50):
        song = songs[songs.loc[:, 'song_id']==recommended_list[i]]
        print('歌曲: %s, 歌手: %s, 专辑: %s, 评分为: %lf' % (song["title"].values[0], song['artist_name'].values[0], song['release'].values[0], recommended_score[i]))
    new_list = []
    for i in range(len(recommended_list)):
        if R.loc[user, recommended_list[i]] == 0:
            new_list.append((recommended_list[i], recommended_score[i]))
    print("==============================================")
    print('推荐评分前50个(不包括以前听过的):')
    for i in range(50):
        song = songs[songs.loc[:, 'song_id'] == new_list[i][0]]
        print('歌曲: %s, 歌手: %s, 专辑: %s, 评分为: %lf' % (song["title"].values[0], song['artist_name'].values[0], song['release'].values[0], new_list[i][1]))


if __name__ == '__main__':
    data = read_data('data/data.csv')
    R = get_R(data)
    K = 50
    learning_rate = 0.01
    beta = 0
    steps = 10
    P, Q = init_P_Q(np.array(R), K)
    P, Q, loss = LFM(np.array(R), P, Q, K, learning_rate=learning_rate, steps=steps, beta=beta)
    # print(loss)
    plot_loss(loss, learning_rate=learning_rate)
    R_hat = P @ Q.T
    R_hat = pd.DataFrame(R_hat, index=R.index, columns=R.columns)
    s = input("请输入需要推荐的用户的用户名:")
    recommended_song(R, R_hat, s)
