import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn
import os

def calculate_mse_mae(y_true, y_pred):
    '''
    Args: lists of GLN matrix
    Return: 
    '''
    y_true_number = np.array(list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_true)))
    y_pred_number = np.array(list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_pred)))
    mse = ((y_pred_number-y_true_number)**2).mean()
    mae = (abs(y_pred_number-y_true_number)).mean()
    return mse, mae


def calculate_gln_accuracy(y_true, y_pred, threshold=0.8):
    'GLN_accuracy is defined as the percentage of samples with prediction residuals < 0.1'
    tp, fp, tn, fn = [0, 0, 0, 0]
    for y1, y2 in zip(y_true, y_pred):
        if abs(y1) >= threshold and abs(y2) >= threshold:
            tp += 1
        elif abs(y1) >= threshold and abs(y2) < threshold:
            fn += 1
        elif abs(y1) < threshold and abs(y2) >= threshold:
            fp += 1
        else:
            tn += 1
    recall = f"{(tp/(tp+fn)):.4f}" if tp+fn != 0 else 'NAN'
    precision = f"{(tp/(tp+fp)):.4f}" if tp+fp !=0 else 'NAN'
    return {'recall': recall, 'precision':precision}


def scatter_true_pred_number(y_true, y_pred, color=None):
    '''
    Test regression model and plot the result
    y_true and y_pred are numbers
    '''
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mse = ((y_pred-y_true)**2).mean()
    mae = (abs(y_pred-y_true)).mean()
    performance_list = [
        calculate_gln_accuracy(y_true, y_pred, threshold=threshold)
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]
    ]
    performance = performance_list[-3]
    rec_list = [
        p['recall']
        for p in performance_list
    ]
    prec_list = [
        p['precision']
        for p in performance_list
    ]
    print('recall:', rec_list)
    print('precision:', prec_list)
    fig1 = plt.figure(dpi=300)
    y_pred = np.where(y_pred>3, 3, y_pred)
    y_pred = np.where(y_pred<-3, -3, y_pred)
    y_true = np.where(y_true>3, 3, y_true)
    y_true = np.where(y_true<-3, -3, y_true)
    threshold = 0.7
    color = []
    for y1, y2 in zip(y_true, y_pred):
        if abs(y1) >= threshold and abs(y2) < threshold:
            color.append('y')
        elif abs(y1) < threshold and abs(y2) >= threshold:
            color.append('y')
        else:
            color.append('b')
    #plt.plot([-4, 4], [-4, 4], ls="--", alpha=0.8, c="lightgrey")    
    plt.xlim((-3.1, 3.1))
    plt.ylim((-3.1, 3.1))
    plt.xticks(np.arange(-3, 4, step=1))
    plt.yticks(np.arange(-3, 4, step=1))
    plt.scatter(x=y_pred, y=y_true, s=8, alpha=0.4, c=color)
    plt.xlabel("Predicted GLN", {'size': 14})
    plt.ylabel("Actual GLN", {'size': 14})
    plt.grid(True)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    result = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\n"
    result += f"recall: {performance['recall']}\nprecision: {performance['precision']}"
    plt.annotate(result, xy=(0.04, 0.75),xycoords='axes fraction', fontsize=13, family='Arial')
    plt.close()

    fig2 = plt.figure(dpi=300)
    plt.plot([0.5, 0.6, 0.7, 0.8, 0.9], rec_list, label='recall')
    plt.plot([0.5, 0.6, 0.7, 0.8, 0.9], prec_list, label='precision')
    plt.xlabel("threshold", {'size': 14})
    plt.close()

    return fig1, fig2


def unit_vector_gln(matrix, index):
    'sum of glns in a L shape at the position of index'
    assert(0 <= index < len(matrix))
    sum = 0
    for i in range(index+1):
        sum += matrix[index][i]
    for i in range(index):
        sum += matrix[i][index]
    return sum


def gln_matrix2number(matrix, start, end):
    'calculate gln number from matrix for the segment from start to end (Left include)'
    assert 0 <= start <= end <= len(matrix)
    matrix = np.squeeze(matrix)
    gln = 0.0 
    for i in range(start, end):
        gln += unit_vector_gln(matrix, i)
    return gln


def compare_number(y_true, y_pred):
    '''
        Compare y_true and y_pred which are gln_matrixes with the shape of (B, L, L)
        y_tue dim: (L-1, L-1)  y_pred dim:(L-1, L-1, 1)
    '''
    print('Comparing number...')
    # Scatter plot whole GLN
    y_true_number = list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_true))
    y_pred_number = list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_pred))

    return scatter_true_pred_number(y_true_number, y_pred_number)


def compare_matrix(y_true, y_pred):
    'Compare matrix'
    print('Comparing matrix...')
    true_unit_list = list(map(lambda x: np.ndarray.flatten(np.array(x)), y_true))
    pred_unit_list = list(map(lambda x: np.ndarray.flatten(np.array(x)), y_pred))
    # Plot true unit glns vs. predicted unit gln
    true_units = np.concatenate(true_unit_list, axis=0)
    pred_units = np.concatenate(pred_unit_list, axis=0)
    pred_units = list(map(lambda x: x if abs(x) <=0.1 else x/abs(x)*0.1, pred_units))
    fig1 = plt.figure(dpi=300)
    plt.xlabel("Predicted GLNs", {'size': 12})
    plt.ylabel("True GLNs", {'size': 12})
    plt.scatter(x=pred_units, y=true_units, s=1, alpha=0.2, linewidths=0, c='blue')
    plt.xlim((-0.11, 0.11))
    plt.ylim((-0.11, 0.11))
    plt.xticks(np.arange(-0.1, 0.15, step=0.05))
    plt.yticks(np.arange(-0.1, 0.15, step=0.05))
    plt.close()
    # Plot predict unit gln errors vs. true whole gln
    fig2 = plt.figure(dpi=300)
    for i, (true, pred) in enumerate(zip(true_unit_list, pred_unit_list)):
        err = pred - true
        err = list(map(lambda x:  x if abs(x) <=0.2 else x/abs(x)*0.2, err))
        plt.scatter(np.linspace(i, i, len(err)), err, s=1, alpha=0.2, linewidths=0, c='blue')
        plt.ylim((-0.21, 0.21))
        plt.yticks(np.arange(-0.2, 0.25, step=0.05))
    plt.xlabel('Samples', {'size': 12})
    plt.ylabel('Predcted unit GLN error', {'size': 12})
    plt.close()

    return fig1, fig2


def draw_single_heatmap(matrix, title, save_path):
    'Draw Heat Map for one matrix'
    matrix = np.squeeze(matrix)
    plt.figure(dpi=300)
    res = seaborn.heatmap(data=matrix, cmap=plt.get_cmap('bwr'), xticklabels=20, yticklabels=20, center=0)
    plt.title(title, fontsize=14)
    # Drawing the frame
    res.axhline(y = 0, color='k',linewidth = 1.5)
    res.axhline(y = matrix.shape[1], color = 'k', linewidth = 1.5)
    res.axvline(x = 0, color = 'k', linewidth = 1.5)
    res.axvline(x = matrix.shape[0], color = 'k', linewidth = 1.5)
    plt.savefig(f'{save_path}.png')
    plt.close()


def draw_heatmap(y_true, y_pred, pids, save_dir):
    'Draw Heat Map'
    print('Drawing Heatmap...')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for pid, matrix in zip(pids, y_true):
        title = f'GLN Heat Map {pid} True'
        path = os.path.join(save_dir, f'heatmap_{pid}_true')
        draw_single_heatmap(matrix, title, path)
    for pid, matrix in zip(pids, y_pred):
        title = f'GLN Heat Map {pid} Pred'
        path = os.path.join(save_dir, f'heatmap_{pid}_pred')
        draw_single_heatmap(matrix, title, path)
    

def compare_loss_residual(loss, y_true, y_pred, ignore_anomaly=False):
    'y_true/pred: matrix'
    y_true_number = list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_true))
    y_pred_number = list(map(lambda x: gln_matrix2number(x, 0, len(x)), y_pred))
    if ignore_anomaly:
        y_true_number_new = []
        y_pred_number_new = []
        loss_new = []
        for true, pred, l in zip(y_true_number, y_pred_number, loss):
            if abs(pred) <= 3:
                y_true_number_new.append(true)
                y_pred_number_new.append(pred)
                loss_new.append(l)
        res = np.absolute(np.array(y_pred_number_new) - np.array(y_true_number_new))
        loss = loss_new
    else:
        res = np.absolute(np.array(y_pred_number) - np.array(y_true_number))
    fig1 = plt.figure(dpi=300)
    plt.scatter(x=res, y=loss, s=8, alpha=0.4, c='b')
    plt.xlabel("Abs(residuals)", {'size': 14})
    plt.ylabel("Backbone Loss", {'size': 14})
    plt.annotate(f'Avg backbone loss: {np.mean(loss):.4f}', xy=(0.15, 0.1),xycoords='axes fraction', fontsize=13)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.close()
    fig2 = plt.figure(dpi=300)
    plt.scatter(x=res, y=loss, s=8, alpha=0.4)
    plt.xlim((0, 2))
    plt.xlabel("Abs(residuals)", {'size': 14})
    plt.ylabel("Backbone Loss", {'size': 14})
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.close()

    return fig1, fig2
    

def scatter_mae_msa(mae, msa_depth):
    fig = plt.figure(dpi=300)
    plt.scatter(x=msa_depth, y=mae, s=8, alpha=0.4, c='b')
    plt.xlabel("Neff", {'size': 14})
    plt.ylabel("Abs(residuals)", {'size': 14})
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    
    return fig


def scatter_2mae(mae_us, mae_base):
    mae_us = np.where(mae_us>3, 3, mae_us)
    mae_base = np.where(mae_base>3, 3, mae_base)

    fig = plt.figure(dpi=300)
    plt.scatter(mae_us, mae_base, s=10, alpha=0.4, c='b')
    plt.xlabel("Abs(residuals) after fine-tuning", {'size': 14})
    plt.ylabel("Abs(residuals) before fine-tuning", {'size': 14})
    plt.plot([-4, 4], [-4, 4], ls="--", alpha=0.8, c="lightgrey")    
    plt.xlim((0, 3.1))
    plt.ylim((0, 3.1))
    plt.xticks(np.arange(0, 4, step=1))
    plt.yticks(np.arange(0, 4, step=1))
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    return fig
