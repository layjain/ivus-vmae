from . import arguments
from . import visualize
from . import augs

import torch
import time

#################################################################################
### General Utils
#################################################################################
def timer_func(func, identifier=""):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {identifier}:{func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


#################################################################################
### Classification Utils
#################################################################################
def get_tp_fp_tn_fn(preds, labels):
    '''
    preds, labels : torch tensors
    '''
    normal_preds = preds[torch.nonzero(torch.where(labels==0, 1, 0))]
    fp = get_item(sum(normal_preds))
    tn = normal_preds.shape[0] - fp

    mal_preds = preds[torch.nonzero(torch.where(labels==1, 1, 0))]
    tp = get_item(sum(mal_preds))
    fn = mal_preds.shape[0] - tp

    diags = {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}

    return diags

def compute_classification_stats(tp, fp, tn, fn, matrix_savepath=None):
    precision = safe_divide(tp , (tp + fp))
    recall = safe_divide(tp , (tp + fn))
    f1 = safe_divide(tp , (tp + 0.5*(fp+fn)))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    balanced_accuracy = 0.5 * (tn/(tn + fp) + tp/(tp+fn))
    confusion_matrix = [[tn, fp], [fn, tp]]
    
    if matrix_savepath is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.matshow(confusion_matrix)

        for (i, j), z in np.ndenumerate(confusion_matrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        plt.ylabel("True Label")
        plt.xlabel("Predicted label")
        plt.title(f'Prec:{precision:.3f}|Rec:{recall:.3f}|F1:{f1:.3f}|Acc:{accuracy:.3f}')
        plt.savefig(matrix_savepath)
        plt.clf(); plt.close()
    return {'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy, 'balanced_accuracy':balanced_accuracy}

def safe_divide(num, den):
    try:
        return num/den
    except ZeroDivisionError as e:
        return None

def dict_merge(d1, d2, assert_unique_keys=True):
    '''
    Simple merge dict operation for non-nested dicts
    '''
    d = d1.copy()
    for k,v in d2.items():
        if (assert_unique_keys) and (k in d):
            raise ValueError(f"Found duplicate key {k}")
        d[k] = v
    return d

def get_item(t):
    '''
    t -- one element torch tensor or float/int
    '''
    if isinstance(t, (float, int)):
        return t
    if isinstance(t, torch.Tensor):
        return t.item()