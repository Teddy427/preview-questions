import tensorcircuit as tc
import tensorflow as tf
import networkx as nx

K = tc.set_backend("tensorflow")

nlayers = 3  # QAOA的层数
ncircuits = 2
# 建图
def dict2graph(d):
    g = nx.to_networkx_graph(d)
    for e in g.edges:
        if not g[e[0]][e[1]].get("weight"):
            g[e[0]][e[1]]["weight"] = 1.0
    return g


example_graph_dict = {
    0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
    1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
    2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
    7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
    6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
    5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
}

example_graph = dict2graph(example_graph_dict)
def QAOAansatz(params, g=example_graph):
    n = len(g.nodes)
    c = tc.Circuit(n)
    # 制备哈密顿量 H_B = X_1 + ... + X_n 的基态
    for i in range(n):
        c.H(i)
    # 核心步骤 PQC，交替执行 H_C 和 H_B 的演化，近似模拟绝热演化过程
    # nlayers 越高模拟效果越好
    for j in range(nlayers):
        # U_j = exp(-iθH_C)
        for e in g.edges:
            c.exp1(
                e[0],
                e[1],
                unitary=tc.gates._zz_matrix,
                theta=g[e[0]][e[1]].get("weight", 1.0) * params[2 * j],
            )
        # V_j = exp(-iθH_B)
        for i in range(n):
            c.rx(i, theta=params[2 * j + 1])
    # 由量子绝热定理保证绝热演化过程后系统一定为 H_C 的基态
    # 但由于是近似演化所以并非一定是
    # 计算测量期望
    loss = 0.0
    for e in g.edges:
        loss += c.expectation_ps(z=[e[0], e[1]])
    return K.real(loss)
# 获取实时损失和梯度
QAOA_vvag = K.jit(tc.backend.vvag(QAOAansatz, argnums=0, vectorized_argnums=0))
# ncircuit 表示并行的电路个数
params = K.implicit_randn(shape=[ncircuits, 2 * nlayers], stddev=0.1)
opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

for i in range(50):
    loss, grads = QAOA_vvag(params, example_graph)
    print(K.numpy(loss))
    params = opt.update(grads, params)  # 梯度下降

print(K.numpy(params)) # 查看参数优化结果
# 助教：才三层而已，没有规律就是最大的规律