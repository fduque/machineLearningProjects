import numpy as np
import matplotlib.pyplot as plt

greyhoundsDogs = 500
labsDogs = 500

grey_height = 28 + 4 * np.random.randn(greyhoundsDogs)
lab_height = 24 + 4 * np.random.randn(labsDogs)

plt.hist([grey_height,lab_height], stacked=True, color=['r','b'])
plt.show()