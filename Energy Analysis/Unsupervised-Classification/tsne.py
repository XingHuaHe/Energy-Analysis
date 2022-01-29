# %%
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %%
feature = np.load('./outputs/test/features.npy')

# %%
X_embedded = TSNE(n_components=2).fit_transform(feature[:,0:10], feature[:,10:11])

# %%
# fig, ax = plt.subplots()

plt.scatter(X_embedded[0:39][:, 0], X_embedded[0:39][:, 1], c='b')  # 2: 39

plt.scatter(X_embedded[39:98, 0], X_embedded[39:98, 1], c='r')  # 1: 59

plt.scatter(X_embedded[98:204, 0], X_embedded[98:204, 1], c='g')  # 0: 106

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()
# %%
