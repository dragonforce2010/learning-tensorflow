import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100, dtype=float)
y1 = 10 * x ** 3 + 8 * x ** 2 + 3 * x + 6
y2 = 3 * x ** 4 - 8 * x ** 2 + 100 * x - 70

plt.figure(num=1, figsize=(8, 8))
plt.plot(x, y1, color='red', linewidth=3.0, linestyle='--')
plt.plot(x, y2, color='green')

plt.figure(num=2, figsize=(8, 8))
plt.plot(x, y2)
plt.show()
