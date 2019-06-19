import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndi

from src.common.dataset import ds


class HeatMap:

    def __init__(self):
        self.screen_resolution = ds.screen_resolution
        self.screen_physical_size = ds.screen_physical_size
        self.screen_distance = ds.screen_distance

    def visual_degree_to_pixels(self, deg):
        """convert visual angle to pixel size on screen"""
        l = self.screen_distance * np.tan(np.deg2rad(deg))
        npx = l * self.screen_resolution[0] / self.screen_physical_size[0]
        npy = l * self.screen_resolution[1] / self.screen_physical_size[1]
        return [npx*0.5, npy*0.5]

    def generate_heatmap(self, x, y, grid_size=None, norm_type=1, img_size=False, smooth_res=False, smooth_degree=1.0,
                         to_plot=False):
        if grid_size is None:
            gridx = np.arange(0, self.screen_resolution[0]+1)
            gridy = np.arange(0, self.screen_resolution[1]+1)
        else:
            gridx = grid_size[0]
            gridy = grid_size[1]

        if img_size:
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=[gridx, gridy],
                                                     range=[[0, gridx], [0, gridy]])
        else:
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=[gridx, gridy],
                                                     range=[[0, self.screen_resolution[0]],
                                                            [0, self.screen_resolution[1]]])
        if smooth_res:
            k = self.visual_degree_to_pixels(smooth_degree)
            heatmap = ndi.gaussian_filter(heatmap, k)

        if norm_type == 1:
            heatmap /= np.sum(heatmap)
        elif norm_type == 2:
            heatmap = np.square(heatmap).astype(np.float32)
            heatmap /= np.sum(heatmap)

        if to_plot:
            fig, arr = plt.subplots(1, 2)
            arr[0].scatter(x[:], y[:])
            arr[0].set_xlim(0, gridx)
            arr[0].set_ylim(0, gridy)
            arr[0].set_aspect('equal')
            arr[1].matshow(heatmap.T, cmap='hot')
            arr[1].set_ylim(0, gridy)
            plt.show()

        return heatmap

hm = HeatMap()


def testIndividualFixSeq():
    fixDir = ds.fixDataPath
    subjectIds = ds.getSubjectIds()
    for si in subjectIds:
        si = 'rh'
        subDir = os.path.join(fixDir, si)
        vfPath = os.path.join(subDir, "viewFixs.npy")
        print(vfPath)
        vDict = np.load(vfPath).item()
        for i in range(100):
            fixs = vDict[i]
            print(fixs)
            hm.generate_heatmap(fixs[:, 3], fixs[:, 4], smooth_res=True, to_plot=True)
            if i == 10:
                break
        break

if __name__ == '__main__':
    # testIndividualFixSeq()
    print(hm.visual_degree_to_pixels(1))