import numpy as np

y = np.load("y_new.npy")

print("Healthy:", np.sum(y==0))
print("Covid:", np.sum(y==1))



X = np.load("X_new.npy")


print("X shape:", X.shape)
print("y shape:", y.shape)