import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

# from line_profiler import LineProfiler
# import line_profiler


all_thresholds = []
sMap = []
const1 = []
const3 = []


def calLevelRatio(i):
    thresh = all_thresholds[i]
    above_indices = np.where(sMap >= thresh)
    above_th = len(above_indices[0])
    tp = const1 * (i + 1)
    fp = const3 * (above_th - i - 1)
    return [tp, fp]


# @profile
def auc(saliency_map, fixation_map, jitter=True, to_plot=False, threading=True):
    """
    AUC - Judd
    Learn to predict where human look, In ICCV 2009
    """
    score = np.nan

    if not len(fixation_map):
        print("No fixation map")
        return score

    if np.isnan(np.sum(saliency_map)):
        print("NAN in saliency map")
        return score

    if jitter:
        saliency_map += np.random.rand(saliency_map.shape[0], saliency_map.shape[1])/10000000

    # normalize saliency map
    min_sa = np.min(saliency_map)
    max_sa = np.max(saliency_map)
    saliency_map = np.subtract(saliency_map, min_sa)
    saliency_map *= 1.0/(max_sa-min_sa)
    global sMap
    sMap = saliency_map

    fix_x, fix_y = np.where(fixation_map > np.finfo(np.float32).eps)
    fix_sa = saliency_map[fix_x, fix_y]  # saliency value at fixation location

    n_fixs = len(fix_x)
    n_pixels = saliency_map.size

    global all_thresholds
    all_thresholds = np.sort(fix_sa)[::-1]
    tp = np.zeros(n_fixs+2, dtype=np.float32)
    fp = np.zeros(n_fixs+2, dtype=np.float32)

    tp[0] = 0.0
    tp[-1] = 1.0
    fp[0] = 0.0
    fp[-1] = 1.0

    global const1, const3
    const1 = 1.0/float(n_fixs)
    const3 = 1.0/float(n_pixels-n_fixs)

    if threading:
        res = Parallel(n_jobs=8)(delayed(calLevelRatio)(l) for l in np.arange(n_fixs))

        res = np.array(res)
        tp[1:-1] = res[:, 0]
        fp[1:-1] = res[:, 1]

    else:
        for i in range(n_fixs):
            thresh = all_thresholds[i]
            above_indices = np.where(saliency_map>=thresh)
            above_th = len(above_indices[0])
            tp[i+1] = const1 * (i+1)
            fp[i+1] = const3 * (above_th-i-1)

    score = np.trapz(tp.T, x=fp.T)
    all_thresholds = np.insert(all_thresholds, 0, 1)
    all_thresholds = np.append(all_thresholds, 0)

    if to_plot:
        f, arr = plt.subplots(1,3)
        arr[0].matshow(saliency_map)
        arr[0].set_title("saliency map")
        arr[1].matshow(fixation_map)
        arr[1].set_title("fixation map")
        arr[2].plot(tp[:], fp[:])
        arr[2].set_title("area under roc curve %.2f"%score)
        plt.show()

    return score, tp, fp, all_thresholds


def normalize_map(map):
    minv = np.min(map)
    maxv = np.max(map)
    res = np.subtract(map, minv)
    res *= 1.0/(maxv - minv)
    return res


def correlation_coefficient(saliency_map1, saliency_map2, to_plot=False):
    if len(saliency_map2.shape) > 1:
        m1 = resize(saliency_map1, saliency_map2.shape, preserve_range=True)
        m1 = m1.astype(np.float)
    else:
        m1 = saliency_map1.astype(np.float)
    m2 = saliency_map2.astype(np.float)

    m1 = normalize_map(m1)
    m2 = normalize_map(m2)

    mean1 = np.mean(m1)
    mean2 = np.mean(m2)

    c = (m1-mean1)*(m2-mean2)
    d = (m1-mean1)**2
    e = (m2-mean2)**2
    score = np.sum(c)/float(np.sqrt(np.sum(d)*np.sum(e)))

    if to_plot:
        f, arr = plt.subplots(1, 2)
        arr[0].matshow(saliency_map1)
        arr[0].set_title("fixation map")
        arr[1].matshow(saliency_map2)
        arr[1].set_title("saliency map")
        f.suptitle("%.2f" % score)
        plt.show()

    return score


def testAUC():
    s = 5
    a = np.random.rand(s, s)
    # b = np.random.uniform(0.0, 1.0, size=(s, s))
    b = copy.deepcopy(a)
    print (a)
    a[np.where(a<0.5)] = 0.0
    b[np.where(b>0.4)] = 1.0
    b[np.where(b<0.4)] = 0.0
    print (a)
    print (b)
    import time
    sTime = time.time()
    # b = 1.0-a
    auc(a, b, jitter=1, to_plot=True, threading=False)
    eTime = time.time()
    print(eTime-sTime)


def testCC():
    a = np.random.randint(0, 10, 5)
    b = np.random.randint(0, 10, 5)
    print (a)
    b = a
    print (b)
    print (correlation_coefficient(a, b))


if __name__ == '__main__':
    # testAUC()
    testCC()