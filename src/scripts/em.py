from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial import distance

from src.common.dataset import ds
from src.common.heatmap import hm


viewFixs = []
recallFixs = []
mappedFixs = []
rad = 25

def get_signature_from_fixation_descriptor(fixs, is_duration_weighted=True):
    """ Get signature from fixation descriptor
        :param fixs: row-wise fixation descriptor, each row x, y, duration, starting timestamp
        :param is_duration_weighted: use duration as point weight
        :return: the signature vector represents the eye movement distribution 
    """
    if np.isnan(np.sum(fixs)):
        print("ERROR!")
        return np.inf

    sig = np.zeros((len(fixs), 3), dtype=np.float32)
    for idx, fix in enumerate(fixs):
        if is_duration_weighted:
            sig[idx, 0] = fix[2]
        else:
            sig[idx, 0] = 1
        sig[idx, 1] = fix[0]
        sig[idx, 2] = fix[1]
    sig[:, 0] = sig[:, 0] / np.sum(sig[:, 0])
    return sig


def reweight_signature(s, weight):
    """ adjust the weight of a signature array
        :param s: input signature with normalized weight
        :param weight: weight adjusting factor
        :return: signature with new weights
    """
    sig = s.copy()
    sig[:, 0] = weight * sig[:, 0]
    return sig


def find_desired_ptr(src, dst, neighbor_weight):
    """ compute the best matching encoding fixation (dst) for each recall fixation (src)
        :param src: recall fixations
        :param dst: encoding fixations
        :param neighbor_weight: weight decay
        :return: signature with new weights
    """
    n = len(dst)

    w = float(neighbor_weight)  # comparatively small

    weights = np.zeros(n)
    iw2 = (w ** 2)
    iw2 = (-1.0) / iw2
    desired_src = []
    for s in src:
        for idx in range(n):
            d2 = np.linalg.norm(s - dst[idx])
            d2 = d2 ** 2
            weights[idx] = np.exp(d2 * iw2)
        if np.sum(weights) > np.finfo(np.float32).eps:
            p = np.dot(dst.T, weights)
            w = 1.0 / np.sum(weights)
            p *= w
        else:  # all points are too far away, don't move
            p = s
        desired_src.append(p.tolist())

    return np.asarray(desired_src).reshape(-1, 2)


def cal_weighted_center(arr, w):
    total_weight = np.sum(w)
    sum_x = np.dot(w, arr[:, 0])
    sum_y = np.dot(w, arr[:, 1])
    return np.array([sum_x / total_weight, sum_y / total_weight])


def estimate_weighted_rigid_transformation(ridx, ptr, desired_ptr, rigid_weight):
    """ estimate local tranformation matrix for a recall fixation 
        :param ridx: index of recall fixation  
        :param ptr: recall fixations 
        :param desired_ptr: closest points in encoding sequence 
        :param rigid_weight: weight parameter controlling the amount of rigid transformation 
        :return: relocated recall fixaions and the tranformation
    """
    n = ptr.shape[0]
    weights = np.zeros(n)
    p = ptr[ridx, :]
    w = float(rigid_weight)
    iw2 = (w ** 2)
    iw2 = (-1.0)/iw2

    for i in range(n):
        d2 = np.linalg.norm(p-ptr[i])
        d2 = d2 ** 2
        # print d2
        weights[i] = np.exp(d2 * iw2)

    # compute weighted concensus location given the desired point location 
    centroid_s = cal_weighted_center(ptr, weights)
    centroid_d = cal_weighted_center(desired_ptr, weights)
    ss = ptr - centroid_s
    dd = desired_ptr - centroid_d

    # compute local rigid transformation 
    tweights = np.tile(weights, (2, 1))
    tmp = np.multiply(tweights.T, dd)
    H = np.dot(ss.T, tmp)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_d.T - np.dot(R, centroid_s.T)
    T = np.identity(3)
    T[0:2, 0:2] = R
    T[0:2, 2] = t
    p_new = np.dot(T, np.array([p[0], p[1], 1]))
    return p_new[:2], T


def asap_with_rigid(recall, view, neighbor_weight=100, rigid_weight=250,
                    max_iteration=50, recall_weight=1.0, to_plot=False):
    """ iteratively compute the elastic matching function
        :param recall: recall fixations 
        :param view: encoding fixations 
        :param others: default parameter values 
        :return: relocated recall fixaions 
    """
    recall = reweight_signature(recall.copy(), recall_weight)
    orig_recall_pts = recall[:, [1, 2]].copy()
    orig_view_pts = view[:, [1, 2]].copy()

    nr = len(orig_recall_pts)
    nv = len(orig_view_pts)
    t0 = np.zeros((nr, 3, 3))
    t0[:] = np.identity(3, dtype=np.float32)

    ptr = np.ones((3, nr))
    ptp = np.ones((3, nv))
    ptr[0:2, :] = np.copy(recall[:, [1, 2]].T)
    ptp[0:2, :] = np.copy(view[:, [1, 2]].T)

    for i in range(max_iteration):
        old_ptr = (ptr[:2, :].T).copy()
        # find corresponding fixation point
        desired_ptr = find_desired_ptr(ptr[:2, :].T, ptp[:2, :].T, neighbor_weight)

        # compute local tranformation for each recall fixation 
        for ridx in range(nr):
            pt_new, T = estimate_weighted_rigid_transformation(ridx, orig_recall_pts, desired_ptr, rigid_weight)
            t0[ridx] = T
            ptr[0:2, ridx] = pt_new
        
        # compute total relocation cost 
        dis = np.linalg.norm(old_ptr - ptr[:2, :].T)

    if to_plot:
        scaler = 1000.0

        plt.scatter(view[:, 1], view[:, 2],
                    s=view[:, 0] * scaler, edgecolors='silver', facecolors='orange', alpha=0.95)
        plt.scatter(orig_recall_pts[:, 0], orig_recall_pts[:, 1],
                    s=recall[:, 0] * scaler, edgecolors='silver', facecolors='green', alpha=0.95)
        plt.scatter(ptr[0,:].T, ptr[1,:].T, s=recall[:, 0] * scaler,
                    edgecolors='silver', facecolors='fuchsia', alpha=1.0)
        # plt.scatter(desired_ptr[:, 0], desired_ptr[:, 1], s=recall[:, 0] * scaler,
        #             edgecolors='silver', facecolors='fuchsia', alpha=1.0)
        plt.ylim([1200, 0])
        plt.xlim([0, 1920])
        plt.show()
    return None, ptr[:2, :].T


def mapRecallToView(sIdx):
    """ map recall fixations to encoding fixations 
        :param sIdx: image index 
        :return: relocated recall fixaions 
    """
    view_fixs = viewFixs[sIdx]
    recall_fixs = recallFixs[sIdx]
    view_sig = get_signature_from_fixation_descriptor(view_fixs[:, [3, 4, 2, 0]])
    recall_sig = get_signature_from_fixation_descriptor(recall_fixs[:, [3, 4, 2, 0]])

    _, pts = asap_with_rigid(recall_sig, view_sig, neighbor_weight=50, rigid_weight=200,
                                         max_iteration=100, recall_weight=1.0)
    matched_recall = recall_fixs.copy()
    matched_recall[:, [3, 4]] = pts
    return matched_recall


def find_neighbors_within_radius(sIdx):
    view_fixs = viewFixs[sIdx][:, [3, 4]]
    mapped_fixs = mappedFixs[sIdx][:, [3, 4]]

    flags = np.zeros((len(view_fixs)), dtype=np.uint8)
    dist = distance.cdist(view_fixs, mapped_fixs)
    for i, row in enumerate(dist):
        n = len(np.where(row < rad)[0])
        if n > 0:
            flags[i] = 1

    filterFlags = flags.tolist()
    filterFlags = np.asarray(filterFlags)

    return [viewFixs[sIdx][np.where(filterFlags == 1)[0]], viewFixs[sIdx][np.where(filterFlags == 0)[0]]]


def main():
    outputPath = '../../res'

    imgNames = ds.getImageNames()

    for imgIdx in range(ds.numImgs):

        print(imgIdx)
        # load image and align it to the coordinate sysmtem used by eye movements
        img = ds.loadImage(imgIdx)
        offset = ds.calImgDisplayOffset(img.shape)

        # load fixations data
        global viewFixs, recallFixs
        viewFixs = np.load(os.path.join(ds.imgDataPath, "%d_viewFixs.npy" % imgIdx))
        recallFixs = np.load(os.path.join(ds.imgDataPath, "%d_recallFixs.npy" % imgIdx))

        # mapping fixations during recall to fixations during encoding
        res = Parallel(n_jobs=8)(delayed(mapRecallToView)(sIdx)
                                 for sIdx in range(ds.numSubjects))
        global mappedFixs
        mappedFixs = res

        # # write out results
        # mappedResPath = os.path.join(ds.mappedRecallFixsPath, '%d_mappedRecallFixs.npy'%imgIdx)
        # np.save(mappedResPath, res)

        # filtering encoding fixations based on relocated recall fixations
        filterRes = Parallel(n_jobs=8)(delayed(find_neighbors_within_radius)(sIdx)
                                 for sIdx in range(ds.numSubjects))
        filterRes = np.asarray(filterRes)
        filterFixs = filterRes[:, 0]
        remFixs = filterRes[:, 1]
        filterFixs = np.vstack(filterFixs).reshape(-1, 6)
        remFixs = np.vstack(remFixs).reshape(-1, 6)

        # # write out results
        # filteredResPath = os.path.join(ds.filteredViewFixsPath, '%d_filteredViewFixs.npy'%imgIdx)
        # np.save(filteredResPath, filterFixs)

        # visualise mapped results 
        viewFixs = np.vstack(viewFixs).reshape(-1, 6)
        recallFixs = np.vstack(recallFixs).reshape(-1, 6)

        # scatter plot of sample points
        # plt.imshow(img, origin='lower')
        # plt.scatter(viewFixs[:, 3]-offset[0], viewFixs[:, 4]-offset[1], c='green')
        # plt.scatter(recallFixs[:, 3]-offset[0], recallFixs[:, 4]-offset[1], c='yellow')
        # plt.scatter(filterFixs[:, 3]-offset[0], filterFixs[:, 4]-offset[1], c='orange')
        # plt.show()

        vHM = hm.generate_heatmap(viewFixs[:, 3]-offset[0], viewFixs[:, 4]-offset[1],
                                grid_size=[img.shape[1], img.shape[0]], img_size=True,
                                smooth_res=True, smooth_degree=1.0, to_plot=False)
        mHM = hm.generate_heatmap(filterFixs[:, 3]-offset[0], filterFixs[:, 4]-offset[1],
                                grid_size=[img.shape[1], img.shape[0]], img_size=True,
                                smooth_res=True, smooth_degree=1.0, to_plot=False)
        remHM = hm.generate_heatmap(remFixs[:, 3]-offset[0], remFixs[:, 4]-offset[1],
                                grid_size=[img.shape[1], img.shape[0]], img_size=True,
                                smooth_res=True, smooth_degree=1.0, to_plot=False)

        fig, arr = plt.subplots(1, 4, figsize=[20, 4])
        for i in range(4):
            arr[i].imshow(img, origin='lower')
            arr[i].set_xticks([])
            arr[i].set_yticks([])
            arr[i].axis('off')

        vHM = np.flip(vHM, 1)
        remHM = np.flip(remHM, 1)
        mHM = np.flip(mHM, 1)
        arr[1].imshow(vHM.T, origin='lower', alpha=0.8, cmap='jet')
        arr[2].imshow(remHM.T, origin='lower', alpha=0.8, cmap='jet')
        arr[3].imshow(mHM.T, origin='lower', alpha=0.8, cmap='jet')

        plt.tight_layout()
        outputFile = os.path.join(outputPath, imgNames[imgIdx].replace('.jpeg', '.pdf'))
        plt.savefig(outputFile, dpi=300)
        # plt.show()
        # break


if __name__ == '__main__':
    main()
