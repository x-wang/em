import numpy as np
import os
from PIL import Image


class Database(object):

    def __init__(self):

        self.clickOrigDataPath = "../../click_exp/data/"

        self.ascDataPath = "../../data/ascData"
        self.fixDataPath = "../../data/fixData"
        self.imgDataPath = "../../data/imgData"
        self.clickDataPath = "../../data/clickData"
        self.stimuliPath = "../../stimuli"
        self.blurStimuliPath = "../../click_exp/stimuliBG"
        self.iccvDataPath = "../../data/iccvData"
        self.resPath = "../../res"
        self.mappedRecallFixsPath = "../../res/mappedRecallFixs"
        self.filteredViewFixsPath = "../../res/filteredViewFixs"

        self.numImgs = 100
        self.numSubjects = 28

        self.screen_resolution = [1920, 1200]
        self.screen_physical_size = [520, 324]
        self.screen_distance = 700

        self.trialSeqTimes = np.array([500, 5000, 1000, 5000, 1500])
        self.clickSeqTimes = np.array([500, 500, 3000, 1000])
        self.trialNumPerGroup = 20

        self.fixFileName = ['viewFixs.npy', 'recallFixs.npy']
        self.clickFileNames = ['rawSamplesDict.npy', 'fixDict.npy', 'clickInfo.npy']

    def _loadSubjectIds(self):
        dataIDs = [s for s in os.listdir(self.ascDataPath) if '.DS_Store' not in s]
        assert len(dataIDs) == self.numSubjects, "Error of data set IDs"
        self.dataIDs = dataIDs

    def loadClickSubjectIds(self):
        ids = [s for s in os.listdir(self.clickOrigDataPath) if 'DS_Store' not in s]
        return ids

    def getSubjectIds(self):
        self._loadSubjectIds()
        return self.dataIDs

    def _loadImgNames(self):
        imgNames = [s for s in os.listdir(self.stimuliPath) if 'DS_' not in s]
        assert len(imgNames) == self.numImgs, "Error of image names"
        self.imgNames = imgNames

    def getImageNames(self):
        self._loadImgNames()
        return self.imgNames

    def loadImage(self, imgIdx):
        if not hasattr(self, 'imgNames'):
            self._loadImgNames()

        imgPath = os.path.join(self.stimuliPath, self.imgNames[imgIdx])
        img = Image.open(imgPath).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGB")
        img = np.array(img) * (1.0 / 255)
        return img

    def loadBlurImage(self, imgIdx):
        if not hasattr(self, 'imgNames'):
            self._loadImgNames()

        imgPath = os.path.join(self.blurStimuliPath, self.imgNames[imgIdx])
        img = Image.open(imgPath).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert("RGB")
        img = np.array(img) * (1.0 / 255)
        return img

    def calImgDisplayOffset(self, imgShape):
        x = int(0.5 * (self.screen_resolution[1] - imgShape[0]))
        y = int(0.5 * (self.screen_resolution[0] - imgShape[1]))
        offWidth = (self.screen_resolution[0] - y - imgShape[1])
        offHight = self.screen_resolution[1] - x - imgShape[0]
        return [offWidth, offHight]

ds = Database()


if __name__ == '__main__':

    print(ds.getSubjectIds())
    print(ds.getImageNames())
