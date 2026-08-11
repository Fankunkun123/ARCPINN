import sys
import os
sys.path.insert(0, '../Utilities/')
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.physics.control.control_plots import matplotlib
import time
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
from itertools import cycle
import pickle
from helper import ScipyOptimizerInterface

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

np.random.seed(1234)
tf.set_random_seed(1234)


# 显示中文字体以及显示负号
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 用来正常显示中文标签

print(tf.__version__)  # 输出TensorFlow的版本号
print(tf.executing_eagerly())  # 输出是否处于Eager Execution模式

class XPINN:
    #  边界点小批量，边界解析解小批量，内部点小批量，内部点全，内部解析解全，边+内数据点，别+内解析解
    def __init__(self, xb0, yb0, xf1, x1f, y1f, xbf, ybf, k, layers1):
        self.x_b0 = xb0[:, 0:1]  # 边界数据点（小批量）
        self.yb0 = yb0  # 边界精确解 小批量

        self.x_f1_stage = xf1[:, 0:1]  # 内部数据点 小批量

        self.x1f = x1f[:, 0:1]  # 内部数据点 全 用于子域分点（不含边界）
        self.y1f = y1f  # 内部精确解 全

        self.xbf = xbf[:, 0:1]  # 边界+内部数据点 全 用于区段长度 子域分点
        self.ybf = ybf  # 边界+内部解析解 全

        self.k = k  # 分段数量

        self.history_MSE = []  # 用于计数
        self.MSE_LBFGS_hist1 = []  # 用于LBFGS优化储存损失
        self.MSE_LBFGS_hist2 = []  # 用于LBFGS优化储存损失 不加权
        self.MSE_LBFGS_hist1_bc = []
        self.MSE_LBFGS_hist1_pde = [[] for _ in range(self.k)]
        self.w_MSE_LBFGS_hist1_bc = []
        self.w_MSE_LBFGS_hist1_pde = [[] for _ in range(self.k)]


        self.layers1 = layers1  # 一个神经网络NN
        # self.weights1, self.biases1, self.A1 = self.initialize_NN(layers1)
        self.weights1, self.biases1 = self.initialize_NN(layers1)

        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        #                                              log_device_placement=True))

        # 构建TensorFlow占位符
        self._build_placeholders()

        # 构建前向传播网络
        self._build_network()

        # 构建自适应权重系数
        self._build_adaptiveRS()

        # 第三阶段使用的权重变量
        self.w_ub_final = tf.Variable(initial_value=1.0, dtype=tf.float64, trainable=False)
        self.w_f_final = [tf.Variable(initial_value=1.0, dtype=tf.float64, trainable=False) for _ in range(self.k)]

        # 定义损失函数
        self._define_losses()

        # 定义优化器
        self._define_optimizers()

        # 初始化会话
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=1)

        # weights1_np = self.sess.run(self.weights1)
        # print("weights1 (as NumPy array):")
        # print(weights1_np)

    def _build_placeholders(self):
        """创建TensorFlow占位符"""
        # 边界条件占位符
        self.x_b0_tf = tf.placeholder(tf.float64, [None, 1], name='x_b0')
        self.yb0_tf = tf.placeholder(tf.float64, [None, 1], name='yb0')

        # 内部点占位符（第一阶段）
        self.x_f1_tf = tf.placeholder(tf.float64, [None, 1], name='x_f1_stage')

        # 内部条件占位符 全
        self.x1f_tf = tf.placeholder(tf.float64, [None, 1], name='x1f')

        # 全量数据占位符 边界+内部
        self.xbf_tf = tf.placeholder(tf.float64, [None, 1], name='xbf')

        # 第二阶段分组占位符（动态创建）
        self.group_placeholders = []
        for i in range(self.k):
            x_ph = tf.placeholder(tf.float64, [None, 1], name=f'x_f{i}')
            self.group_placeholders.append(x_ph)

        # 第三阶段分组占位符（动态创建）

    def _build_network(self):
        """构建前向传播网络"""
        # 边界预测
        self.ub1_pred = self.net_u1(self.x_b0_tf)  # 边界网络预测 小批量用于Loss训练 绘图使用
        # PDE残差预测
        self.f_pred1 = self.net_f(self.x_f1_tf)  # 第一阶段 小批量 用于adam训练
        self.uf1_pred = self.net_u1(self.x_f1_tf)  # 内部预测 绘图使用

        # 全量残差计算（用于分类）
        self.ubf_pred = self.net_u1(self.xbf_tf)
        self.res = tf.square(self.ubf_pred)

    def _build_adaptiveRS(self):
        ''' 自适应系数 '''
        a, b, c = 1, 1, 2
        # 边界自适应损失权重
        self.w_ub1 = tf.Variable(a, dtype=tf.float64) ** b  # loss1初始项
        self.sw_ub1 = tf.log(self.w_ub1 ** c)  # loss1 初始项 对数转换
        self.w_ub12 = 0.5 * tf.exp(-self.sw_ub1)  # loss1 对数转换之后的 初始 残差项的 损失系数
        # pde自适应损失权重
        d = 1
        self.w_uf = {}
        self.sw_f = {}
        self.w_f = {}
        self.sum1 = self.sw_ub1 ** d
        for i in range(0, self.k):
            w_uf_name = f'w_uf{i}'
            sw_f_name = f'sw_f{i}'
            w_f_name = f"w_f{i}2"
            self.w_uf[w_uf_name] = tf.Variable(a, dtype=tf.float64) ** b
            self.sw_f[sw_f_name] = tf.log(self.w_uf[w_uf_name] ** c)
            self.w_f[w_f_name] = 0.5 * tf.exp(-self.sw_f[sw_f_name])
            self.sum1 += self.sw_f[sw_f_name] ** d

    def _define_losses(self):
        """定义两阶段损失函数"""
        # 第一阶段损失：边界条件 + PDE残差
        self.loss1 = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))  # L_bc
        self.loss1 += tf.reduce_mean(tf.square(self.f_pred1))  # L_pde
        self.bc_loss_1 = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        self.pde_loss_1 = tf.reduce_mean(tf.square(self.f_pred1))

        # 第二阶段损失：边界条件 + 多组PDE残差
        self.loss2 = (self.w_ub12 / N_ub) * tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred)) + tf.abs(self.sum1)**2 / (N_uf1 * (self.k+1)) #tf.abs(self.sum1) # L_bc
        f1_preds = []
        for i in range(self.k):
            x_tf = self.group_placeholders[i]
            f1_pred = self.net_f(x_tf)
            f1_preds.append(f1_pred)
            sample_size = tf.shape(f1_preds[i])[0]
            # print('sample_size=:', sample_size)
            w_f_name = f"w_f{i}2"
            self.loss2 += (self.w_f[w_f_name] / tf.cast(sample_size, dtype=tf.float64)) * tf.reduce_mean(tf.square(f1_preds[i]))

        # 不加权总损失
        self.loss22 = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))  # tf.abs(self.sum1) # L_bc
        for i in range(self.k):
            self.loss22 += tf.reduce_mean(tf.square(f1_preds[i]))

        # 各损失函数
        self.bc_loss = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        self.w_bc_loss = (self.w_ub12 / N_ub) * tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))

        self.pde_loss = []
        self.w_pde_loss = []
        for i in range(self.k):
            x_tf = self.group_placeholders[i]
            f2_pred = self.net_f(x_tf)

            pde_loss_i = tf.reduce_mean(tf.square(f2_pred))
            self.pde_loss.append(pde_loss_i)

            sample_size = tf.cast(tf.shape(f2_pred)[0], dtype=tf.float64)
            print('sample_size=:', sample_size)
            w_f_name = f"w_f{i}2"
            w_pde_loss_i = (self.w_f[w_f_name] / tf.cast(sample_size, dtype=tf.float64)) * pde_loss_i
            self.w_pde_loss.append(w_pde_loss_i)

        # 第三阶段损失：L-BFGS边界条件 + 多组PDE残差
        # self.loss3 = self.w_ub1_hist1[-1] * tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        # for i in range(self.K):
        #     self.loss3 += self.w_f1_hist1[i][-1] * tf.reduce_mean(tf.square(f1_preds[i]))

        self.loss3 = (self.w_ub_final/ N_ub) * tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        for i in range(self.k):
            x_tf = self.group_placeholders[i]
            f1_pred = self.net_f(x_tf)
            f1_preds.append(f1_pred)
            sample_size = tf.shape(f1_preds[i])[0]
            print('sample_size=:', sample_size)
            self.loss3 += (self.w_f_final[i] / tf.cast(sample_size, dtype=tf.float64)) * tf.reduce_mean(tf.square(f1_preds[i]))

        # 不加权总损失
        self.loss33 = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        for i in range(self.k):
            self.loss33 += tf.reduce_mean(tf.square(f1_preds[i]))

        #  各损失函数
        self.bc_loss_3 = tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))
        self.w_bc_loss_3 = (self.w_ub_final/ N_ub) * tf.reduce_mean(tf.square(self.yb0_tf - self.ub1_pred))

        self.pde_loss_3 = []
        self.w_pde_loss_3 = []
        for i in range(self.k):
            x_tf = self.group_placeholders[i]
            f3_pred = self.net_f(x_tf)

            pde_loss_3i = tf.reduce_mean(tf.square(f3_pred))
            self.pde_loss_3.append(pde_loss_3i)

            sample_size = tf.cast(tf.shape(f3_pred)[0], dtype=tf.float64)
            print('sample_size=:', sample_size)
            w_pde_loss_3i = (self.w_f_final[i] / tf.cast(sample_size, dtype=tf.float64)) * pde_loss_3i
            self.w_pde_loss_3.append(w_pde_loss_3i)

    def _define_optimizers(self):
        """定义优化器"""
        # 第一阶段优化器（较大学习率）
        self.optimizer_stage1 = tf.train.AdamOptimizer(learning_rate=5e-3)
        self.train_op1 = self.optimizer_stage1.minimize(self.loss1)

        # 第二阶段优化器（较小学习率）
        self.optimizer_stage2 = tf.train.AdamOptimizer(learning_rate=5e-3)
        self.train_op2 = self.optimizer_stage2.minimize(self.loss2)

        # 第三阶段优化器（较小学习率）

        self.optimizer_LBFGS = ScipyOptimizerInterface(self.loss3, method='L-BFGS-B',
                                                   options={'maxiter': 30000,
                                                            'maxfun': 30000,
                                                            'maxcor': 50,
                                                            'maxls': 20,
                                                            'ftol': 1e-50,
                                                            'gtol': 1e-50})  # 'ftol': 1e-50 * np.finfo(float).eps

    # 使用 Xavier 初始化方法初始化权重矩阵可以有效地提高神经网络的性能。偏置向量则是用来增加各层神经元的灵活性的。
    # #使用正则化参数 a 可以通过 L2 正则化控制网络的复杂度，从而防止过拟合。
    def initialize_NN(self, layers):  # 生成权重，偏置
        weights = []
        biases = []
        # A = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            w = self.xavier_init(size=[layers[l], layers[l + 1]])  # self.xavier_init函数生成了每一层神经网络的权重矩阵
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            # a = tf.Variable(0.1, dtype=tf.float64)
            weights.append(w)
            biases.append(b)
            # A.append(a)
        # return weights, biases, A
        return weights, biases

    def xavier_init(self, size):  # 生成权重
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))  # 计算了标准差 xavier_stddev
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)

    # def neural_net_tanh(self, X, weights, biases, A):
    def neural_net_tanh(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            # H = tf.tanh(10*A[l] * tf.add(tf.matmul(H, w), b))
            H = tf.tanh(tf.add(tf.matmul(H, w), b))
        w = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, w), b)
        return Y

    # def neural_net_sin(self, X, weights, biases, A):
    def neural_net_sin(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            w = weights[l]
            b = biases[l]
            # H = tf.sin(10*A[l] * tf.add(tf.matmul(H, w), b))
            H = tf.sin(tf.add(tf.matmul(H, w), b))
        w = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, w), b)
        return Y

    def net_u1(self, x):
        # u = self.neural_net_sin(tf.concat([x, y], 1), self.weights1, self.biases1, self.A1)
        u = self.neural_net_sin(tf.concat([x], 1), self.weights1, self.biases1)
        return u

    def f(self, x):
        pi = tf.constant(np.pi, dtype=tf.float64)
        u = 0.5 * tf.sin(omega1 * pi * x) * tf.cos(omega2 * pi * x) + tf.tanh(20 * x)
        du_xx = -0.5 * pi ** 2 * omega1 ** 2 * tf.sin(pi * omega1 * x) * tf.cos(
            pi * omega2 * x) - 1.0 * pi ** 2 * omega1 * omega2 * tf.sin(pi * omega2 * x) * tf.cos(
            pi * omega1 * x) - 0.5 * pi ** 2 * omega2 ** 2 * tf.sin(pi * omega1 * x) * tf.cos(pi * omega2 * x) - 20 * (
                        40 - 40 * tf.tanh(20 * x) ** 2) * tf.tanh(20 * x)
        f = du_xx - la * u
        return f

    def net_f(self, x):
        # Sub-Net1
        u1 = self.net_u1(x)
        u1_x = tf.gradients(u1, x)[0]
        u1_xx = tf.gradients(u1_x, x)[0]

        f1 = u1_xx - la * u1 - self.f(x)
        return f1

    def residual_classification(self):
        """残差分类模块（使用GMM聚类）"""
        # 计算全量残差
        feed_dict = {self.xbf_tf: self.xbf}
        residuals = self.sess.run(self.res, feed_dict)

        # 数据标准化 确保每个特征具有0均值和单位方差
        scaler = StandardScaler()
        scaled_res = scaler.fit_transform(residuals)

        # GMM聚类
        gmm = GaussianMixture(n_components=self.k, random_state=42)
        labels = gmm.fit_predict(scaled_res)

        # 按标签分组数据
        grouped_data = []
        q_grouped_data = []
        cluster_sizes = [np.sum(labels == i) for i in range(self.k)]  # 每个簇的大小
        # print(f'cluster_sizes:{cluster_sizes}')

        for i in range(self.k):
            mask = (labels == i)  # 布尔掩码 表示对应标签的数据
            group_x = self.xbf[mask, 0:1]
            group_res = residuals[mask]

            # 根据簇大小占总样本的比例确定小批量数据的数量
            samples_to_pick = int(cluster_sizes[i] * 1)

            indices = np.random.choice(len(group_res), samples_to_pick, replace=False)
            # print(f'indices:{indices}:')
            # 使用选定的indices从小批量数据中选取
            sampled_group_x = group_x[indices]
            sampled_group_res = group_res[indices]

            grouped_data.append((sampled_group_x, sampled_group_res))
            q_grouped_data.append((group_x, group_res))
            # print(f'group_x:{i}:')
            # for idx, data in enumerate(group_x):
            #     print(f'Index: {idx}, Data: {data}')
            # print(f'group_t:{i}:')
            # for idx, data in enumerate(group_t):
            #     print(f'Index: {idx}, Data: {data}')
            # print(f'group_res:{i}:')
            # for idx, data in enumerate(group_res):
            #     print(f'Index: {idx}, Data: {data}')

        return grouped_data, q_grouped_data

    def callback(self, loss, bc_loss_3, pde_loss_3, w_bc_loss_3, w_pde_loss_3, loss33):
        self.history_MSE.append(loss)
        step = len(self.history_MSE)
        if step % 100 == 0:
            loss1 = loss  # 加权总损失
            loss11 = loss33 # 未加权总损失
            bc_loss = bc_loss_3
            pde_loss = pde_loss_3
            w_bc_loss = w_bc_loss_3
            w_pde_loss = w_pde_loss_3
            self.MSE_LBFGS_hist1.append(loss1)  # 加权总损失
            self.MSE_LBFGS_hist2.append(loss11)  #  未加权总损失

            self.MSE_LBFGS_hist1_bc.append(bc_loss)
            for i, loss in enumerate(pde_loss):
                self.MSE_LBFGS_hist1_pde[i].append(loss)

            self.w_MSE_LBFGS_hist1_bc.append(w_bc_loss)
            for i, loss in enumerate(w_pde_loss):
                self.w_MSE_LBFGS_hist1_pde[i].append(loss)
            print()
            print('It: %d,' 'loss1: %.2e '% (step, loss1))
            print('bc_loss: %.2e,' ' w_bc_loss: %.2e,' % (bc_loss, w_bc_loss), end='')
            print()
            for i, pde_loss_i in enumerate(pde_loss):
                print('pde_loss%d: %.2e,' % (i, pde_loss_i), end='')
            print()
            for i, w_pde_loss_i in enumerate(w_pde_loss):
                print('w_pde_loss%d: %.2e,' % (i, w_pde_loss_i), end='')
            print()

            np.savez(f'数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_MSE_LBFGS_hist.npz',
            self.MSE_LBFGS_hist1, self.MSE_LBFGS_hist1_bc, np.array(self.MSE_LBFGS_hist1_pde), self.w_MSE_LBFGS_hist1_bc, np.array(self.w_MSE_LBFGS_hist1_pde), self.MSE_LBFGS_hist2)
        return self.MSE_LBFGS_hist1, self.MSE_LBFGS_hist1_bc, self.MSE_LBFGS_hist1_pde, self.w_MSE_LBFGS_hist1_bc, self.w_MSE_LBFGS_hist1_pde, self.MSE_LBFGS_hist2

    def train(self, nIter_stage1, nIter_stage2, nIter_stage3, X_star1, X_star2, u_exactb1, u_exactf1, max_cycles, tol1=1e-5, tol2=5e-3):

        MSE_history1 = []
        MSE_history2 = []  #第二阶段不加权损失
        w_ub0_history1 = []
        w_f0_history1 = [[] for _ in range(self.k)]

        w_ub1_history1 = []
        w_f1_history1 = [[] for _ in range(self.k)]

        # 第一阶段 各损失函数
        MSE_bc_history1_1 = []
        MSE_pde_history1_1 = []

        # 第二阶段 各损失函数
        MSE_bc_history1_2 = []
        MSE_pde_history1_2 = [[] for _ in range(self.k)]
        w_MSE_bc_history1_2 = []
        w_MSE_pde_history1_2 = [[] for _ in range(self.k)]

        weights1_history1 = []
        biases1_history1 = []

        cycle = 0
        # 初始化全局变量
        self.sess.run(tf.global_variables_initializer())

        # 第一阶段训练
        print("\n========    开始第一阶段训练   =========")
        feed_dict1 = {self.x_b0_tf: self.x_b0, self.x_f1_tf: self.x_f1_stage, self.yb0_tf: self.yb0}

        start_time = time.time()
        for it in range(nIter_stage1+1):
            self.sess.run(self.train_op1, feed_dict1)  # 训练操作
            if it % 100 == 0:
                loss1_value = self.sess.run(self.loss1, feed_dict1)

                bc_loss = self.sess.run(self.bc_loss_1, feed_dict1)
                pde_loss = self.sess.run(self.pde_loss_1, feed_dict1)
                elapsed = time.time() - start_time
                # 打印损失信息
                print('It: %d, Loss1: %.2e' % (it, loss1_value), end=' ')
                print(f"Time: {elapsed:.2f}s")
                print(f'BC Loss: {bc_loss:.2e}, PDE Loss: {pde_loss:.2e}')

                u_pred1, u_pred2 = model.predict(X_star1, X_star2)
                u_pred = np.concatenate([u_pred1, u_pred2], axis=0)  # 边+内 预测（全）
                u_exact = np.concatenate([u_exactb1, u_exactf1], axis=0)  # 边+内 解析解（全）

                l2_error = 1 / len(u_pred) * np.linalg.norm(u_exact - u_pred, 2)
                l2_err = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
                max_err = np.linalg.norm(u_exact - u_pred, np.inf)

                print('平均绝对err: %.2e, 二范数err: %.2e, 无穷范数err: %.2e' %
                      (l2_error, l2_err, max_err))
                print()

                MSE_history1.append(loss1_value)
                MSE_bc_history1_1.append(bc_loss)
                MSE_pde_history1_1.append(pde_loss)

        while cycle < max_cycles:
            cycle += 1

            # 步骤1：残差分类
            print("\n=== 进行残差分类 ===")
            grouped_data, _ = self.residual_classification()
            print('cycle=:', cycle)
            # model.fig(cycle)

            feed_dict2 = feed_dict1.copy()
            for i in range(self.k):
                feed_dict2[self.group_placeholders[i]] = grouped_data[i][0]

            # 步骤2：第二阶段训练
            print(f"\n=== 开始第二阶段训练 === 循环训练 {cycle}/{max_cycles} ===")
            start_time = time.time()
            for it in range(nIter_stage2+1):
                self.sess.run(self.train_op2, feed_dict2)
                # 每100步输出训练信息
                if it % 100 == 0:
                    loss2_value = self.sess.run(self.loss2, feed_dict2)
                    loss22_value = self.sess.run(self.loss22, feed_dict2)

                    bc_loss = self.sess.run(self.bc_loss, feed_dict2)
                    pde_loss = self.sess.run(self.pde_loss, feed_dict2)
                    w_bc_loss = self.sess.run(self.w_bc_loss, feed_dict2)
                    w_pde_loss = self.sess.run(self.w_pde_loss, feed_dict2)

                    # print('pde_loss', len(pde_loss))
                    # print('pde_loss', pde_loss)
                    # print('w_pde_loss', len(w_pde_loss))
                    # print('w_pde_loss', w_pde_loss)

                    w_ub1 = self.sess.run(self.w_ub12, feed_dict2)
                    w_f_values = [self.sess.run(self.w_f[f"w_f{i}2"], feed_dict2) for i in range(self.k)]
                    sum_loss1 = w_ub1 + sum(w_f_values)

                    elapsed = time.time() - start_time

                    print(f"Iter {it:5d}, Loss2: {loss2_value:.2e}, Time: {elapsed:.2f}s")
                    print(' bc_loss: %.2e,' ' w_bc_loss: %.2e,'% (bc_loss,  w_bc_loss), end='')
                    print()
                    for i, pde_loss_i in enumerate(pde_loss):
                        print(' pde_loss%d: %.2e,' % (i, pde_loss_i), end='')
                    print()
                    for i, w_pde_loss_i in enumerate(w_pde_loss):
                        print(' w_pde_loss%d: %.2e,' % (i, w_pde_loss_i), end='')
                    print()

                    # 打印自适应权重信息 未归一化
                    print(' 0w_ub1: %.2e,' % (w_ub1), end='')
                    for i, w_f in enumerate(w_f_values):
                        print(' 0w_f%d: %.2e,' % (i, w_f), end='')
                    print()

                    u_pred21, u_pred22 = model.predict(X_star1, X_star2)
                    u_pred2 = np.concatenate([u_pred21, u_pred22], axis=0)  # 边+内 预测（全）
                    u_exact = np.concatenate([u_exactb1, u_exactf1], axis=0)  # 边+内 解析解（全）

                    l2_error = 1 / len(u_pred2) * np.linalg.norm(u_exact - u_pred2, 2)
                    l2_err = np.linalg.norm(u_exact - u_pred2, 2) / np.linalg.norm(u_exact, 2)
                    max_err = np.linalg.norm(u_exact - u_pred2, np.inf)

                    print('平均绝对err: %.2e, 二范数err: %.2e, 无穷范数err: %.2e' %
                          (l2_error, l2_err, max_err))
                    print()

                    MSE_history1.append(loss2_value)
                    MSE_history2.append(loss22_value)
                    # 归一化
                    w_ub0_history1.append(w_ub1 / sum_loss1)
                    for i, w_f in enumerate(w_f_values):
                        w_f0_history1[i].append(w_f / sum_loss1)
                    # 未归一化
                    w_ub1_history1.append(w_ub1)
                    for i, w_f in enumerate(w_f_values):
                        w_f1_history1[i].append(w_f)

                    # 各项损失函数
                    MSE_bc_history1_2.append(bc_loss)
                    for i, loss in enumerate(pde_loss):
                        MSE_pde_history1_2[i].append(loss)

                    w_MSE_bc_history1_2.append(w_bc_loss)
                    for i, loss in enumerate(w_pde_loss):
                        w_MSE_pde_history1_2[i].append(loss)

        w_ub_final_value = w_ub1_history1[-1] if w_ub1_history1 else 1.0
        w_f_final_values = [w_f1_history1[i][-1] if w_f1_history1[i] else 1.0 for i in range(self.k)]
        # 更新TensorFlow变量
        assign_ops = [self.w_ub_final.assign(w_ub_final_value)]
        for i in range(self.k):
            assign_ops.append(self.w_f_final[i].assign(w_f_final_values[i]))
        self.sess.run(assign_ops)

        # 步骤1：残差分类
        print("\n=== 进行残差分类 ===")
        grouped_data, _ = self.residual_classification()

        # model.fig(cycle+1)
        feed_dict3 = feed_dict1.copy()
        for i in range(self.k):
            feed_dict3[self.group_placeholders[i]] = grouped_data[i][0]

        # 步骤2：第三阶段训练
        print(f"\n=== 开始第三阶段训练 ===")
        self.optimizer_LBFGS.minimize(self.sess, feed_dict=feed_dict3, fetches=[self.loss3, self.bc_loss_3, self.pde_loss_3, self.w_bc_loss_3, self.w_pde_loss_3, self.loss33], loss_callback=self.callback)

        return MSE_history1, w_ub0_history1, w_f0_history1, w_ub1_history1, w_f1_history1, MSE_bc_history1_1, MSE_pde_history1_1, MSE_bc_history1_2, MSE_pde_history1_2, w_MSE_bc_history1_2, w_MSE_pde_history1_2, MSE_history2
    # 存模型
    def save_model(self):
        model_file = f"./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_test.ckpt"
        self.saver.save(self.sess, model_file)

    # 读取模型
    def load_model(self, X_star1, X_star2, u_exactb, u_exact1):
        model_file = f'./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_test.ckpt'
        self.saver.restore(self.sess, model_file)
        load_b1 = self.sess.run(self.ub1_pred, {self.x_b0_tf: X_star1[:, 0:1]})
        load_f1 = self.sess.run(self.uf1_pred, {self.x_f1_tf: X_star2[:, 0:1]})
        return load_b1, load_f1

    # 输入数据 X_star1 和 X_star2，然后通过执行计算图中的操作，获取预测结果
    def predict(self, X_star1, X_star2):
        u_star1 = self.sess.run(self.ub1_pred, {self.x_b0_tf: X_star1[:, 0:1]})
        u_star2 = self.sess.run(self.uf1_pred, {self.x_f1_tf: X_star2[:, 0:1]})
        return u_star1, u_star2

    def exact(self, x):
        pi = np.pi
        u = 0.5 * np.sin(omega1 * pi * x) * np.cos(omega2 * pi * x) + np.tanh(20 * x)
        return u

    def fig(self, cycle1):
        # 用于绘制 GMM-分类 散点图
        _, q_grouped_data = model.residual_classification()
        uf_x_segments = [[] for _ in range(k)]
        uf_y_segments = [[] for _ in range(k)]
        for i in range(k):
            uf_x_segments[i].append(q_grouped_data[i][0])
            uf_y_segments[i].append(q_grouped_data[i][1])
        # 保存数据到文件
        # for i in range(k+1):
        with open(
                f'./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k, xh} 学习率={xxl} 循环={cycle1-1} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_gmm-分类.pkl',
                'wb') as f:
            pickle.dump((uf_x_segments, uf_y_segments), f)
        # 转换为NumPy数组
        for i in range(k):
            uf_x_segments[i] = np.array(uf_x_segments[i])
            uf_y_segments[i] = np.array(uf_y_segments[i])
        for i in range(k):
            uf_x_segments[i] = uf_x_segments[i].flatten()[:, None]
            uf_y_segments[i] = uf_y_segments[i].flatten()[:, None]

        # 创建一个颜色和标记样式的无限循环
        colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
        markers = cycle(['o'])

        # 创建一个新的图形
        fig, ax = plt.subplots()
        # 设置坐标轴范围从0到1，并且确保没有额外的空白区域
        ax.set_xlim(np.min(x_total), np.max(x_total))

        # 移除顶部和右侧的边框
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # 加粗底部和左侧的边框（即坐标轴）
        ax.spines['top'].set_linewidth(0)
        ax.spines['bottom'].set_linewidth(0)
        ax.spines['left'].set_linewidth(0)
        ax.spines['right'].set_linewidth(0)

        color_cycle = cycle(colors)

        # 绘制每个子域的数据点
        for i in range(k):
            # 重置颜色循环器，使其从头开始
            if i == 0:
                color_cycle = cycle(colors)
            color = next(color_cycle)
            marker = next(markers)  # 因为只有一个标记样式，所以直接使用第一个元素
            # 检查子域是否包含数据点
            if uf_x_segments[i].size > 0:
                ax.scatter(uf_x_segments[i], self.exact(uf_x_segments[i]), c=color, marker=marker, label=f'Segment {i}', s=20)
        # # 绘制边界点
        # if ub_x is not None and ub_t is not None:
        #     ax.scatter(ub_x, ub_t, c='black', marker='x', s=90, label='Boundary Points')
        # 设置坐标轴标签
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        plt.savefig(
            f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k, xh} 循环={cycle1-1} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_区域划分.png',
            dpi=1000)


# 用于检测当前脚本是否在主程序中运行
if __name__ == "__main__":

    la = 100
    omega1 = 7
    omega2 = 5

    k = 4  # 高斯分组数
    xh = 1  # 循环次数

    numb = 500
    xxl = "0.005,0.005"

    step1 = '1w'
    step2 = '3w'
    step3 = '3w'

    N_uf = 99
    N_ub = 2
    N_uf1 = 499  # 用于第一步adam训练 小批量

    layers1 = [1, 20, 20, 1]
    data = np.load(f'./数据+结果/数据/w={omega1,omega2} 训练点={numb}_shuju.npz')

    ub_x = data['arr_0']
    ub_y = data['arr_1']

    uf_x = data['arr_2']
    uf_y = data['arr_3']

    x_total = data['arr_4']
    u_total = data['arr_5']

    X_train_b = ub_x.flatten()[:, None]
    X_train_f = uf_x.flatten()[:, None]
    X_total = x_total.flatten()[:, None]  # 边界+内部
    # print('X_train_b:')
    # for idx, data in enumerate(X_train_b):
    #     print(f'Index: {idx}, Data: {data}')
    # print('X_train_f:')
    # for idx, data in enumerate(X_train_f):
    #     print(f'Index: {idx}, Data: {data}')
    # print('X_total:')
    # for idx, data in enumerate(X_total):
    #     print(f'Index: {idx}, Data: {data}')

    y1_exactb = u_total[0:len(ub_x), ].flatten()[:, None]
    y1_exactf = u_total[len(ub_x):len(ub_x) + len(uf_x), ].flatten()[:, None]
    y1_total = u_total.flatten()[:, None]
    # print('y1_total:')
    # for idx, data in enumerate(y1_total):
    #     print(f'Index: {idx}, Data: {data}')

    # X_star1 = np.hstack((ub_x.flatten()[:, None]))
    # X_star2 = np.hstack((uf_x.flatten()[:, None]))

    X_star1 = np.vstack((ub_x.flatten()[:, None]))
    X_star2 = np.vstack((uf_x.flatten()[:, None]))

    idx1 = np.random.choice(y1_exactb.shape[0], N_ub, replace=False)
    idx2 = np.random.choice(y1_exactf.shape[0], N_uf1, replace=False)

    X_train_1b = X_train_b[idx1, :]
    u_exactb1 = y1_exactb[idx1, :]
    X_train_1f = X_train_f[idx2, :]

    # XPINN模型
    model = XPINN(X_train_1b, u_exactb1, X_train_1f, X_train_f, y1_exactf, X_total, y1_total, k, layers1)

    nIter_stage1 = 10000
    nIter_stage2 = 30000
    nIter_stage3 = 30000
    start_time = time.time()

    # 单次调用 model.train() 并获取所有需要保存的历史数据
    (MSE_hist1,
     w_ub0_hist1, w_f0_hist1, w_ub1_hist1, w_f1_hist1,
     MSE_bc_hist1_1, MSE_pde_hist1_1, MSE_bc_hist1_2, MSE_pde_hist1_2, w_MSE_bc_hist1_2, w_MSE_pde_hist1_2,
     MSE_hist2) = (
        model.train(nIter_stage1, nIter_stage2, nIter_stage3, X_star1, X_star2, y1_exactb, y1_exactf, xh))

    # 构建文件路径和名称
    npz_file_path = f"./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_MSE_hist.npz"
    # pkl_file_path = f"./数据+结果/数据/λ={lambd} 子域={k} 迭代={step1,step2,step3} 激活函数={jhhs} max={maxcor,maxls} 学习率={xxl} 训练点={num,numb} 小批量={N_uf,N_ub}_weights_biases_hist.pkl"

    # 保存所有历史数据（除了权重和偏置）到一个 .npz 文件
    np.savez(npz_file_path, MSE_hist1,
             w_ub0_hist1, np.array(w_f0_hist1), w_ub1_hist1, np.array(w_f1_hist1),
             MSE_bc_hist1_1, MSE_pde_hist1_1, MSE_bc_hist1_2, np.array(MSE_pde_hist1_2), w_MSE_bc_hist1_2, np.array(w_MSE_pde_hist1_2), MSE_hist2)
    # 使用pickle保存权重和偏置历史数据到一个 .pkl 文件
    # with open(pkl_file_path, 'wb') as pkl_file:
    #     pickle.dump({'weights': weights1_hist1, 'biases': biases1_hist1}, pkl_file)

    print()
    print('................................')
    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    ################################################################
    model.save_model()
    ################################################################
    ####                        加载模型                         ####
    ################################################################
    load_file = f"./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_MSE_hist.npz"
    data = np.load(load_file, allow_pickle=True)
    MSE_hist1 = data['arr_0']   # 加权总损失
    w_ub0_hist1 = data['arr_1']
    w_f0_hist1 = data['arr_2']
    w_ub1_hist1 = data['arr_3']
    w_f1_hist1 = data['arr_4']
    MSE_bc_hist1_1 = data['arr_5']
    MSE_pde_hist1_1 = data['arr_6']
    MSE_bc_hist1_2 = data['arr_7']
    MSE_pde_hist1_2 = data['arr_8']
    w_MSE_bc_hist1_2 = data['arr_9']
    w_MSE_pde_hist1_2 = data['arr_10']
    MSE_hist2 = data['arr_11']  # 第二阶段未加权总损失

    load_file1 = f"./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k, xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_MSE_LBFGS_hist.npz"
    data1 = np.load(load_file1, allow_pickle=True)
    MSE_LBFGS_hist1 = data1['arr_0']  # 加权总损失
    MSE_LBFGS_hist1_bc = data1['arr_1']
    MSE_LBFGS_hist1_pde = data1['arr_2']
    w_MSE_LBFGS_hist1_bc = data1['arr_3']
    w_MSE_LBFGS_hist1_pde = data1['arr_4']
    MSE_LBFGS_hist2 = data1['arr_5']  # 未加权总损失

    MSE_adam_LBFGS_hist = np.hstack((MSE_hist1, MSE_LBFGS_hist1)) # 加权总损失
    MSE_adam_LBFGS_hist1 = np.hstack((MSE_hist1[0:nIter_stage1//100], MSE_hist2, MSE_LBFGS_hist2)) # 未加权总损失

    actual_iterations = len(MSE_hist1) * 100
    actual_iterations1 = len(MSE_adam_LBFGS_hist) * 100

    print('actual_iterations=:', actual_iterations)
    print('actual_iterations1=:', actual_iterations1)

    # with open(f'./数据+结果/数据/PINN(u2)/λ={lambd} 子域={k} 迭代={step1,step2,step3} 激活函数={jhhs} max={maxcor,maxls} 学习率={xxl} 训练点={num,numb} 小批量={N_uf,N_ub}_weights_biases_hist.pkl', 'rb') as f:
    #     loaded_values = pickle.load(f)
    # weights1_hist1 = loaded_values['weights']
    # biases1_hist1 = loaded_values['biases']
    # values = [[] for _ in range(2*(len(layers1)-1))]
    # for i in range(len(layers1)-1):
    #     values[i] = weights1_hist1[0][i]  # 直接添加列表
    # for i in range(len(layers1)-1):
    #     values[len(layers1) - 1 + i] = biases1_hist1[0][i]  # 直接添加列表
    #
    # # 保存数据到文件
    # with open(f'./数据+结果/数据/PINN(u2)/w={omega} 激活函数={jhhs} 学习率={xxl} 训练点={num, numb} 小批量={N_uf, N_ub}_权重-偏置.pkl', 'wb') as f:
    #     pickle.dump(values, f)

    # with open(f'./数据+结果/数据/PINN(u2)/w={omega} 激活函数={jhhs} 学习率={xxl} 训练点={num, numb} 小批量={N_uf, N_ub}_权重-偏置.pkl', 'rb') as f:
    #     loaded_values = pickle.load(f)


    # load_file1 = f'./数据+结果/数据/λ={lambd} 子域={k} 迭代={step1,step2,step3} 激活函数={jhhs} max={maxcor,maxls} 学习率={xxl} 训练点={num,numb} 小批量={N_uf,N_ub}_MSE_LBFGS_hist.npz'
    # data1 = np.load(load_file1)
    # MSE_LBFGS_hist1 = data1['arr_0']
    #
    # MSE_adam_LBFGS_hist = np.hstack((MSE_hist1, MSE_LBFGS_hist1))
    #
    # actual_iterations = len(MSE_hist1) * 500
    # actual_iterations1 = len(MSE_adam_LBFGS_hist) * 500
    #
    load_b1, load_f1 = model.load_model(X_star1, X_star2, y1_exactb, y1_exactf)
    u_load_y1 = np.hstack((load_b1.flatten(), load_f1.flatten()))

    u_pred1 = load_b1
    u_pred2 = load_f1
    u_pred_u1 = u_load_y1
    u_exact = np.hstack((y1_exactb.flatten(), y1_exactf.flatten()))

    l2_error = 1/len(u_exact) * np.linalg.norm(u_exact-u_pred_u1, 2)
    l2_err = np.linalg.norm(u_exact-u_pred_u1, 2)/np.linalg.norm(u_exact, 2)
    max_err = np.linalg.norm(u_exact-u_pred_u1, np.inf)
    np.savez(
        f'./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_predict.npz',
        u_pred_u1, l2_error, l2_err, max_err)
    print('平均绝对Error u: %.2e' % (l2_error))
    print('二范数Error u: %.2e' % (l2_err))
    print('无穷范数Error u: %.2e' % (max_err))

    # 用于绘制 GMM-分类 散点图
    _, q_grouped_data = model.residual_classification()
    uf_x_segments = [[] for _ in range(k)]
    uf_y_segments = [[] for _ in range(k)]
    for i in range(k):
        uf_x_segments[i].append(q_grouped_data[i][0])
        uf_y_segments[i].append(q_grouped_data[i][1])
    # 保存数据到文件
    with open(
            f'./数据+结果/数据/λ={la} w={omega1,omega2} 子域={k,xh} 循环={xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_gmm-分类.pkl',
            'wb') as f:
        pickle.dump((uf_x_segments, uf_y_segments), f)
    # 转换为NumPy数组
    for i in range(k):
        uf_x_segments[i] = np.array(uf_x_segments[i])
        uf_y_segments[i] = np.array(uf_y_segments[i])
    for i in range(k):
        uf_x_segments[i] = uf_x_segments[i].flatten()[:, None]
        uf_y_segments[i] = uf_y_segments[i].flatten()[:, None]

    '''################################################################
       ####                     loss下降曲线                       ####
       ################################################################'''
    fig1 = plt.figure()
    plt.plot(range(1, actual_iterations1 + 1, 100), MSE_adam_LBFGS_hist, 'r-', linewidth=2.2, label='Loss')
    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_LossDeclineCurve.png', dpi=1000)

    # 加权-未加权总损失
    fig1 = plt.figure()
    plt.plot(range(1, actual_iterations1 + 1, 100), MSE_adam_LBFGS_hist, 'r-', linewidth=2.2, label='w_loss')
    plt.plot(range(1, actual_iterations1 - 100 + 1, 100), MSE_adam_LBFGS_hist1, 'b-', linewidth=2.2, label='loss')
    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1, omega2} 子域={k, xh} 学习率={xxl} 迭代次数={step1, step2, step3} 训练点={numb} 小批量={N_uf}_加权-未加权总损失.png',
        dpi=1000)

    '''################################################################
           ####                     第一阶段 loss下降曲线                       ####
           ################################################################'''
    fig1 = plt.figure()
    plt.plot(range(1, nIter_stage1+101, 100), MSE_bc_hist1_1, 'r-', linewidth=2.2, label='bc_loss')
    plt.plot(range(1, nIter_stage1+101, 100), MSE_pde_hist1_1, 'b-', linewidth=2.2, label='pde_loss')
    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1, omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1, step2, step3} 训练点={numb} 小批量={N_uf}_第一阶段-各项损失函数.png',
        dpi=1000)

    '''################################################################
          ####                 第二阶段  不带权重Loss                       ####
          ################################################################'''
    fig9 = plt.figure()
    plt.plot(range(1, nIter_stage2+101, 100), MSE_bc_hist1_2, 'c-', linewidth=2.2, label='Loss_BC')
    # 获取颜色列表，以便为不同的线分配不同的颜色
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    color_index = 0
    linestyles = ['-', '--', '-.', ':']
    linestyle_index = 0

    # 遍历所有的系数历史并绘制
    for i, hist in enumerate(MSE_pde_hist1_2):
        # 如果颜色列表用完了，就循环使用
        if color_index >= len(colors):
            color_index = 0
        # 如果线型列表用完了，就循环使用
        if linestyle_index >= len(linestyles):
            linestyle_index = 0
        # 选择当前的颜色
        color = colors[color_index]
        linestyle = linestyles[linestyle_index]
        # 绘制当前系数的历史
        plt.plot(range(1, nIter_stage2+101, 100), hist, color=color, linestyle=linestyle, linewidth=2.2,
                 label=f'Loss_PDE{i}')
        # 更新颜色和线型索引
        color_index += 1
        linestyle_index += 1

    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_第二阶段-不带权重各项Loss.png',
        dpi=1000)

    '''################################################################
              ####               第二节段  带自适应损失权重Loss                       ####
              ################################################################'''
    fig9 = plt.figure()
    plt.plot(range(1, nIter_stage2+101, 100), w_MSE_bc_hist1_2, 'c-', linewidth=2.2, label='Loss_BC')
    # 获取颜色列表，以便为不同的线分配不同的颜色
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    color_index = 0
    linestyles = ['-', '--', '-.', ':']
    linestyle_index = 0

    # 遍历所有的系数历史并绘制
    for i, hist in enumerate(w_MSE_pde_hist1_2):
        # 如果颜色列表用完了，就循环使用
        if color_index >= len(colors):
            color_index = 0
        # 如果线型列表用完了，就循环使用
        if linestyle_index >= len(linestyles):
            linestyle_index = 0
        # 选择当前的颜色
        color = colors[color_index]
        linestyle = linestyles[linestyle_index]
        # 绘制当前系数的历史
        plt.plot(range(1, nIter_stage2+101, 100), hist, color=color, linestyle=linestyle, linewidth=2.2,
                 label=f'Loss_PDE{i}')
        # 更新颜色和线型索引
        color_index += 1
        linestyle_index += 1

    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_第二阶段-带自适应损失权重各项Loss.png',
        dpi=1000)

    '''################################################################
             ####                 第三阶段  不带权重Loss                       ####
             ################################################################'''
    fig9 = plt.figure()
    plt.plot(range(1, nIter_stage3 + 1, 100), MSE_LBFGS_hist1_bc, 'c-', linewidth=2.2, label='Loss_BC')
    # 获取颜色列表，以便为不同的线分配不同的颜色
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    color_index = 0
    linestyles = ['-', '--', '-.', ':']
    linestyle_index = 0

    # 遍历所有的系数历史并绘制
    for i, hist in enumerate(MSE_LBFGS_hist1_pde):
        # 如果颜色列表用完了，就循环使用
        if color_index >= len(colors):
            color_index = 0
        # 如果线型列表用完了，就循环使用
        if linestyle_index >= len(linestyles):
            linestyle_index = 0
        # 选择当前的颜色
        color = colors[color_index]
        linestyle = linestyles[linestyle_index]
        # 绘制当前系数的历史
        plt.plot(range(1, nIter_stage3 + 1, 100), hist, color=color, linestyle=linestyle, linewidth=2.2,
                 label=f'Loss_PDE{i}')
        # 更新颜色和线型索引
        color_index += 1
        linestyle_index += 1

    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1, omega2} 子域={k, xh} 学习率={xxl} 迭代次数={step1, step2, step3} 训练点={numb} 小批量={N_uf}_第三阶段-不带权重各项Loss.png',
        dpi=1000)

    '''################################################################
              ####               第三节段  带自适应损失权重Loss                       ####
              ################################################################'''
    fig9 = plt.figure()
    plt.plot(range(1, nIter_stage3 + 1, 100), w_MSE_LBFGS_hist1_bc, 'c-', linewidth=2.2, label='Loss_BC')
    # 获取颜色列表，以便为不同的线分配不同的颜色
    colors = ['b', 'g', 'r', 'm', 'y', 'k']
    color_index = 0
    linestyles = ['-', '--', '-.', ':']
    linestyle_index = 0

    # 遍历所有的系数历史并绘制
    for i, hist in enumerate(w_MSE_LBFGS_hist1_pde):
        # 如果颜色列表用完了，就循环使用
        if color_index >= len(colors):
            color_index = 0
        # 如果线型列表用完了，就循环使用
        if linestyle_index >= len(linestyles):
            linestyle_index = 0
        # 选择当前的颜色
        color = colors[color_index]
        linestyle = linestyles[linestyle_index]
        # 绘制当前系数的历史
        plt.plot(range(1, nIter_stage3 + 1, 100), hist, color=color, linestyle=linestyle, linewidth=2.2,
                 label=f'Loss_PDE{i}')
        # 更新颜色和线型索引
        color_index += 1
        linestyle_index += 1

    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(
        f'./数据+结果/结果/λ={la} w={omega1, omega2} 子域={k, xh} 学习率={xxl} 迭代次数={step1, step2, step3} 训练点={numb} 小批量={N_uf}_第三阶段-带自适应损失权重各项Loss.png',
        dpi=1000)

    '''################################################################
        ####                     LOSS1自适应权重曲线图                       ####
        ################################################################'''
    fig2 = plt.figure()
    # 获取颜色列表，以便为不同的线分配不同的颜色
    colors = ['g', 'r', 'c', 'm', 'y', 'k']
    color_index = 0
    linestyles = ['-', '--', '-.', ':']
    linestyle_index = 0

    plt.plot(range(1, actual_iterations - 100 - nIter_stage1 + 1 , 100), w_ub0_hist1, 'b', linewidth=2.2,label='w_b1')
    # 遍历所有的系数历史并绘制
    for i, hist in enumerate(w_f0_hist1):
        # 如果颜色列表用完了，就循环使用
        if color_index >= len(colors):
            color_index = 0
        # 如果线型列表用完了，就循环使用
        if linestyle_index >= len(linestyles):
            linestyle_index = 0
        # 选择当前的颜色
        color = colors[color_index]
        linestyle = linestyles[linestyle_index]
        # 绘制当前系数的历史
        plt.plot(range(1, actual_iterations - 100 - nIter_stage1 + 1, 100), hist, color=color, linestyle=linestyle, linewidth=2.2, label=f'w_f{i}')
        # 更新颜色和线型索引
        color_index += 1
        linestyle_index += 1
    plt.xlabel('$\#$ iterations', fontsize=14)
    plt.ylabel('Coefficient', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_自适应权重曲线.png', dpi=1000)

    "##################  精确——预测  ###################"
    fig4 = plt.figure()

    # 对x和y进行排序
    sorted_points = sorted(zip(x_total, y1_total))
    x_total_sorted, y1_total_sorted = zip(*sorted_points)
    sorted_points_1 = sorted(zip(x_total, u_pred_u1.flatten()))
    x_total_sorted_1, u_pred_y1_sorted = zip(*sorted_points_1)

    plt.plot(x_total_sorted, y1_total_sorted, 'k', linewidth=2.2, label='exact')
    plt.plot(x_total_sorted_1, u_pred_y1_sorted, 'r', linestyle='--', linewidth=2.2, label='pred')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('u', fontsize=14)
    plt.legend()
    plt.savefig(f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_u-exact-Pred.png', dpi=1000)

    # "################## 误差 ###################"
    # fig5 = plt.figure()
    #
    # # 对x和y进行排序
    # sorted_points = sorted(zip(x_total, u_pred_u1.flatten() - y1_total.flatten()))
    # x_total_sorted, u_pred_err_sorted = zip(*sorted_points)
    #
    # plt.plot(x_total_sorted, u_pred_err_sorted, 'b', linewidth=2.2, label='y1')
    # # plt.plot(x_total, u_pred_y1.flatten() - y1_total, 'b', linewidth=2, label='y1')
    # plt.xlabel('x', fontsize=14)
    # plt.ylabel('u-err', fontsize=14)
    # plt.savefig(f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb} 小批量={N_uf}_u-err.png', dpi=1000)

    # ''' 子域 '''
    # # 创建一个颜色和标记样式的无限循环
    # colors = cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    # markers = cycle(['o'])
    # # 创建一个新的图形
    # fig, ax = plt.subplots()
    #
    # # 设置坐标轴范围从0到1，并且确保没有额外的空白区域
    # ax.set_xlim(np.min(x_total), np.max(x_total))
    # # 移除顶部和右侧的边框
    # # ax.spines['top'].set_visible(False)
    # # ax.spines['right'].set_visible(False)
    # # 加粗底部和左侧的边框（即坐标轴）
    # ax.spines['top'].set_linewidth(0)
    # ax.spines['bottom'].set_linewidth(0)
    # ax.spines['left'].set_linewidth(0)
    # ax.spines['right'].set_linewidth(0)
    #
    # # 绘制每个子域的数据点
    # for i in range(k):
    #     color = next(colors)
    #     marker = next(markers)
    #     # 检查子域是否包含数据点
    #     if uf_x_segments[i].size > 0:
    #         ax.scatter(uf_x_segments[i], model.exact(uf_x_segments[i]), c=color, marker=marker, label=f'Segment {i}', s=20)
    # # # 绘制边界点
    # # if ub_x is not None and ub_t is not None:
    # #     ax.scatter(ub_x, ub_t, c='black', marker='x', s=90, label='Boundary Points')
    # # 设置坐标轴标签
    # ax.set_xlabel('x', fontsize=14)
    # ax.set_ylabel('y', fontsize=14)
    # # ax.legend()
    # plt.savefig(
    #     f'./数据+结果/结果/λ={la} w={omega1,omega2} 子域={k,xh} 循环={xh} 学习率={xxl} 迭代次数={step1,step2,step3} 训练点={numb}_区域划分.png',
    #     dpi=1000)

    plt.show()