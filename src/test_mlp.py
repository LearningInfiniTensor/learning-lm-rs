import numpy as np

# 参数设置
seq_len = 4  # 假设序列长度
d = 2  # 假设输出维度
di = 3  # 假设中间维度
eps = 1e-5

# 初始化张量
residual = np.ones((seq_len, d), dtype=np.float32)
print("Initial residual:", residual)

hidden_states = np.zeros((seq_len, d), dtype=np.float32)
gate_buf = np.zeros((seq_len, di), dtype=np.float32)
up_buf = np.zeros((seq_len, di), dtype=np.float32)

# 权重
w_up = np.array([[0.1, 0.2],  # shape (di, d)
                 [0.3, 0.4],
                 [0.5, 0.6]], dtype=np.float32)
w_down = np.array([[0.1, 0.2, 0.3],  # shape (d, di)
                   [0.4, 0.5, 0.6]], dtype=np.float32)
w_gate = np.array([[0.1, 0.2],  # shape (di, d)
                   [0.3, 0.4],
                   [0.5, 0.6]], dtype=np.float32)
rms_w = np.array([1.0, 1.0], dtype=np.float32)  # shape (d)

# 自定义 RMS Normalization 函数
def rms_norm(tensor, rms_w, eps):
    norm = np.sqrt(np.mean(tensor ** 2, axis=0, keepdims=True) + eps)
    normalized = (tensor / norm) * rms_w
    print("RMS Norm:", normalized)
    return normalized

# 计算流程
hidden_states = rms_norm(residual, rms_w, eps)
print("Hidden states after RMS norm:", hidden_states)

gate = np.dot(hidden_states, w_gate.T)  # hidden @ gate_weight.T
print("Gate:", gate)

up = np.dot(hidden_states, w_up.T)      # hidden @ up_weight.T
print("Up:", up)

# itermediate = gate * sigmoid(gate) * up 
def silu(up, gate):
    activated = up * (1 / (1 + np.exp(-gate))) * gate
    print("Intermediate (Silu output):", activated)
    return activated

# 计算 intermediate
intermediate = silu(up, gate)  # gate * sigmoid(gate) * up
output = np.dot(intermediate, w_down.T)  # output = intermediate @ down_weight.T
print("Output:", output)

residual = output + residual  # residual = output + residual
print("Final residual:", residual)

