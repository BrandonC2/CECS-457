import numpy  as np
import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 10, 100)
y = 2 * x + 1 + torch.randn(100) * 2 # target y

'''
ones_col = torch.ones(x.size(0), 1)

x = x.unsqueeze(1) # what is this

X = torch.cat((x, ones_col), dim=1)

alpha = torch.linalg.lstsq(X, y).solution

x, y = alpha[0].item(), alpha[1].item()
'''
plt.plot(x, y, label='Data Points', color='blue', alpha=0.6)


# .ones_like() adds as much 1's as the size of x
X = torch.stack([x, torch.ones_like(x)], dim = 1)
# what is stack? 
# takes multiple tensors and stacks them along a new dimension
# all tensors must have the same shape

print(X[0:10, :]) # 10 rows and all columns

a,b = torch.linalg.lstsq(X, y.view(100, 1)).solution

print(a.item(), b.item())

plt.plot(x, a*x+b,  label='Fitted Line', color='red')

plt.show()