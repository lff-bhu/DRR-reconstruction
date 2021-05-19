import numpy as np
import os
import math
import os.path as osp
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import skimage.metrics
import matplotlib.pyplot as plt

def save_record(output_path, epoch, train_loss, val_loss):
    filename = osp.join(output_path, 'loss_record.npz')
    np.savez(filename, epoch=epoch, train_loss=train_loss, val_loss=val_loss)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getErrorMetrics(im_pred, im_gt, mask=None):
    im_pred = np.array(im_pred).astype(np.float)
    im_gt = np.array(im_gt).astype(np.float)
    # sanity check
    assert(im_pred.flatten().shape==im_gt.flatten().shape)
    
    # PSNR
    #psnr_pred = compare_psnr
    psnr_pred = skimage.metrics.peak_signal_noise_ratio(im_gt, im_pred)
    # SSIM
    #ssim_pred = compare_ssim(X=im_gt, Y=im_pred)
    ssim_pred= skimage.metrics.structural_similarity(im_gt, im_pred)
    # MSE
    mse_pred = mean_squared_error(im_gt.flatten(),im_pred.flatten())
    # MAE
    mae_pred = mean_absolute_error(im_gt.flatten(), im_pred.flatten())
    # RMSE
    rmse_pred = math.sqrt(mse_pred)
    print("Compare prediction with groundtruth CT:")
    print('mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
          .format(mae_pred=mae_pred, mse_pred=mse_pred, rmse_pred=rmse_pred, psnr_pred=psnr_pred, ssim_pred=ssim_pred))
    return mae_pred, mse_pred, rmse_pred, psnr_pred, ssim_pred
def getPred(data, idx, test_idx=None):
    pred = data[idx,:,:,:]
    return pred
def imageSave(pred, groundtruth, plane, save_path):
    seq = range(pred.shape[plane])
    for slice_idx in seq:
        if plane == 0:
            pd = pred[slice_idx, :, :]
            gt = groundtruth[slice_idx, :, :]
        elif plane == 1:
            pd = pred[:, slice_idx, :]
            gt = groundtruth[:, slice_idx, :]
        elif plane == 2:
            pd = pred[:, :, slice_idx]
            gt = groundtruth[:, :, slice_idx]
        else:
            assert False
        f = plt.figure()
        f.add_subplot(1,3,1)
        plt.imshow(pd, interpolation='none', cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        f.add_subplot(1,3,2)
        plt.imshow(gt, interpolation='none', cmap='gray')
        plt.title('Groundtruth')
        plt.axis('off')
        f.add_subplot(1,3,3)
        plt.imshow(gt-pd, interpolation='none', cmap='gray')
        plt.title('Difference image')
        plt.axis('off')
        f.savefig(os.path.join(save_path, 'Plane_{}_ImageSlice_{}.png'.format(plane, slice_idx+1)))
        plt.close()