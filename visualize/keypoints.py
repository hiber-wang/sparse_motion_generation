import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

from functools import reduce
from operator import add


__all__ = ['plot2D', 'plot3D']


openpose_chain = [
    [0, 1, 3],
    [0, 2, 4],
    [0, 17, 5, 7, 9],
    [0, 17, 6, 8, 10],
    [0, 17, 11, 13, 15, 19, 20],
    [0, 17, 12, 14, 16, 22, 23],
    [15, 21],
    [16, 24],
    [9, 25, 26, 27, 28],
    [9, 29, 30, 31, 32],
    [9, 33, 34, 35, 36],
    [9, 37, 38, 39, 40],
    [9, 41, 42, 43, 44],
    [10, 45, 46, 47, 48],
    [10, 49, 50, 51, 52],
    [10, 53, 54, 55, 56],
    [10, 57, 58, 59, 60],
    [10, 61, 62, 63, 64]
]



def to_pairs(ls):
    if len(ls) < 2:
        return []
    return [(ls[0], ls[1])] + to_pairs(ls[1:])


def plot2D(jnts, kinematic_chain, savepath='demo.gif'):
    fig, ax = plt.subplots()
    links = reduce(add, [to_pairs(link) for link in kinematic_chain])
    ax.axis('off')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    def update(frame):
        for collection in ax.collections:
            collection.remove()
        lines = []
        for link in links:
            s, t = link
            lines.append(np.array([jnts[frame, s], jnts[frame, t]]))
        collection = LineCollection(lines, color='darkblue')
        ax.add_collection(collection)

    ani = FuncAnimation(fig, update, frames=len(jnts), repeat=True)
    ani.save(savepath, writer='ffmpeg', fps=30, dpi=300)


def plot3D(jnts, kinematic_chain, radius=4, figsize=(10, 10), savepath='demo.gif'):
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        # fig.suptitle(title, fontsize=20)
        # ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = jnts.copy().reshape(len(jnts), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'yellow','green','red', 'blue',  
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
             'darkred', 'darkred','darkred','darkred','darkred']
    frame_number = data.shape[0]
    trajec = data[:, 0, [0, 2]]

    def update(index):
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1]-trajec[index, 1], linewidth=1.0,
                      color='blue')
        
        
        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/30, repeat=False)
    ani.save(savepath, writer='ffmpeg')