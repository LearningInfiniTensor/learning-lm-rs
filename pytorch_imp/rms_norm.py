import torch
def rms_norm(matrix, w, epsilon):
    print("-----------------------rms_norm------------------------")
    print("input = ")
    print(matrix)
    print(w)

    last_dim_size = matrix.size(-1)

    squares = torch.sum(matrix ** 2 , dim=-1, keepdim=True) / last_dim_size + epsilon
    print("squares = ")
    print(squares)

    norm = torch.sqrt(squares)
    print("norm = ")
    print(norm)

    y = (w * matrix) / norm
    print("result = ")
    print(y)
    print("-------------------------------------------------------")


# Same with the given test.
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
w = torch.tensor([1.0, 2.0], dtype=torch.float32)
epsilon = 1e-6 
rms_norm(matrix, w, epsilon)

# A random example
matrix = torch.randn(2, 2, 3, 3, dtype=torch.float32)
w = torch.tensor([10., 12., 8.], dtype=torch.float32)
epsilon = 1e-6 
rms_norm(matrix, w, epsilon)