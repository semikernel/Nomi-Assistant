import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # 启用交互模式
fig, ax = plt.subplots()
x = np.arange(0, 1024)
line, = ax.plot(x, np.random.rand(1024))

def callback(indata, frames, time, status):
    line.set_ydata(indata[:,0])  # 更新波形图
    fig.canvas.draw_idle()

with sd.InputStream(callback=callback, blocksize=1024):
    plt.show()
    while True:
        plt.pause(0.01)