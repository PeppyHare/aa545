import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

## Prepare initial data
a = 20
d = 20
tm0 = 10
t_m = tm0 * np.pi/180
def phi(t):
    return np.pi/4 + (6 + 0.1*t) * t
def r(t):
    return d/2 * (1 - phi(t)/phi(t_m))
def Fp(x,y,t):
    return -1/np.sqrt((x + r(t)*np.cos(phi(t)))**2 + (y + r(t)*np.sin(phi(t)))**2)                                                            \
           -1/np.sqrt((x - r(t)*np.cos(phi(t)))**2 + (y - r(t)*np.sin(phi(t)))**2)     
X = np.arange(-1*a, a)
Y = np.arange(-1*a, a)
x,y=np.meshgrid(X,Y)

## generate image data
imageList = []
for t in range(tm0):
    imageList.append(Fp(x,y,t*np.pi/180))

## Create animation and video from 2D images
fig = plt.figure(figsize=(10,10))
ims = []
for i in range(len(imageList)):
    im = plt.pcolormesh(imageList[i], animated = True)
    ims.append([im])
    
ani1 = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=2000)
plt.colorbar()
plt.show()
t1=time.time()
ani1.to_html5_video()
t2=time.time()
print("2D image to video took: ", t2 - t1)

## Create animation and video from 3D images
t1 = time.time()
fig = plt.figure()
ax = Axes3D(fig)
ims = []
for i in range(len(imageList)):
    im = ax.plot_surface(x,y,imageList[i], antialiased=False, animated=True)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=2000)
# plt.show()
t2 = time.time()
print("3D animation creation took: ", t2 - t1)
ani.to_html5_video()
t3=time.time()
print("3D animation to video took:", t3 - t2)





# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# fig = plt.figure()
# ax = fig.add_subplot(111)

# # I like to position my colorbars this way, but you don't have to
# div = make_axes_locatable(ax)
# cax = div.append_axes('right', '5%', '5%')

# def f(x, y):
#     return np.exp(x) + np.sin(y)

# x = np.linspace(0, 1, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# # This is now a list of arrays rather than a list of artists
# frames = []
# for i in range(10):
#     x       += 1
#     curVals  = f(x, y)
#     frames.append(curVals)

# cv0 = frames[0]
# im = ax.imshow(cv0, origin='lower') # Here make an AxesImage rather than contour
# cb = fig.colorbar(im, cax=cax)
# tx = ax.set_title('Frame 0')

# def animate(i):
#     arr = frames[i]
#     vmax     = np.max(arr)
#     vmin     = np.min(arr)
#     im.set_data(arr)
#     im.set_clim(vmin, vmax)
#     tx.set_text('Frame {0}'.format(i))
#     # In this version you don't have to do anything to the colorbar,
#     # it updates itself when the mappable it watches (im) changes

# ani = animation.FuncAnimation(fig, animate, frames=10)

# plt.show()







# from numpy import random
# from matplotlib import animation
# import matplotlib.pyplot as plt

# img_lst_1 = [random.random((368, 1232)) for i in range(10)]  # Test data
# img_lst_2 = [random.random((368, 1232)) for i in range(10)]  # Test data

# fig, (ax1, ax2) = plt.subplots(2, 1)
# frames = []  # store generated images
# for i in range(len(img_lst_1)):

#     img1 = ax1.imshow(img_lst_1[i], animated=True)
#     img2 = ax2.imshow(img_lst_2[i], cmap="gray", animated=True)

#     frames.append([img1, img2])

# ani = animation.ArtistAnimation(
#     fig, frames, interval=50, blit=True, repeat_delay=1000
# )
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig, ax = plt.subplots()


# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# # ims is a list of lists, each row is a list of artists to draw in the
# # current frame; here we are just animating one artist, the image, in
# # each frame
# ims = []
# for i in range(60):
#     x += np.pi / 15.
#     y += np.pi / 20.
#     im = ax.imshow(f(x, y), animated=True)
#     if i == 0:
#         ax.imshow(f(x, y))  # show an initial one first
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)

# # To save the animation, use e.g.
# #
# # ani.save("movie.mp4")
# #
# # or
# #
# # writer = animation.FFMpegWriter(
# #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# # ani.save("movie.mp4", writer=writer)

# plt.show()
