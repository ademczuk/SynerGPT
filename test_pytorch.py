# test_pytorch.py
import torch

if torch.cuda.is_available():
    print("PyTorch can use your GPU!")
else:
    print("PyTorch will use CPU.")

tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(2, 3)
result = tensor_a + tensor_b 
print(result)
