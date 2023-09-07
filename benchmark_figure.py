import numpy as np
import matplotlib.pyplot as plt

# Matrix sizes
N = np.array([64,128,256,512,1024])

# time in [ms]
T = np.array([ # (10,5)
    [0.0083,0.084,9.59,8.51,71.5],
    [0.0087,0.056,0.45,6.613,53.7],
    [0.141,0.081,9.57,143.4,1162.6],
    [0.0103,0.054,0.45,4.38,39.78],
    [0.0499,0.078,1.58,25.39,218.7],
    [0.0096,0.050,0.27,2.25,17.68],
    [0.023,0.198,1.66,18.9,148],
    [0.033,0.293,1.59,12.91,108.4],
    [0.017,0.118,0.839,6.46,36.45],
    [0.010,0.072,0.49,3.87,31.25]
])

methods = {
    "Standard": 0,
    "Transposed": 1,
    "Block": 2,
    "Block transposed": 3,
    "Block cached": 4,
    "Block cached transposed": 5,
    "Vectorized": 6,
    "Vectorized transposed": 7,
    "Vectorized transposed V2": 8,
    "Vectorized optimal": 9
}

selected_method = [
    "Standard",
    "Transposed",
    "Block",
    "Block transposed",
    "Block cached",
    "Block cached transposed",
    "Vectorized",
    "Vectorized transposed",
    "Vectorized transposed V2",
    "Vectorized optimal"
]


for method_name in selected_method:
    idx = methods[method_name]
    plt.plot(N,T[idx,:], "o-", label=method_name)

plt.xticks(N)
plt.xlabel("Matrix size N$\\times$N", fontsize=14)
plt.ylabel("t [ms]", fontsize=14)
plt.title("C++ matrix multiplication benchmark", fontsize=20)
plt.grid()
plt.legend(frameon=True, loc="upper left")
plt.show()