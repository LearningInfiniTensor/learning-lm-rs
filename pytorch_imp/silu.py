import torch

def silu(x, y):
    print("-----------------------silu-------------------------")
    print("input x = ")
    print(x)
    print("input y = ")
    print(y)
    sigmoid_x = torch.sigmoid(x)
    y = sigmoid_x * x * y

    print("result = ")
    print(y)
    print("----------------------------------------------------")

# Same with the given test.
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 3.0, 4.0])
silu(x, y)

# Same with the given test.
x = torch.randn(8, dtype=torch.float32)
y = torch.randn(8, dtype=torch.float32)
silu(x, y)