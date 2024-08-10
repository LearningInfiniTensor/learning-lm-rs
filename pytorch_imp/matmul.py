import torch

def matmul_trans(alpha, beta, A, B, C):
    print("-----------------------matmul-------------------------")
    # 创建矩阵 A 和 B，形状相同
    print("input A = ")
    print(A)
    print("input B = ")
    print(B)
    print("input C = ")
    print(C)

    ABT = torch.matmul(A, B.T)
    C = alpha * ABT + beta * C

    print("C = ")
    print(C)
    print("-------------------------------------------------------")


# Same with the given test.
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
B = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
C = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
alpha = 1.0
beta = 1.0

matmul_trans(alpha, beta, A, B, C)

# A random case
m, k, n = 2, 3, 4
A = torch.randn(m, k, dtype=torch.float32)
B = torch.randn(n, k, dtype=torch.float32)
C = torch.randn(m, n, dtype=torch.float32)
matmul_trans(alpha, beta, A, B, C)