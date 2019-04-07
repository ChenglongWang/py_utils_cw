import os, time, yaml, json
from termcolor import colored

def Print(*message, color=None, on_color=None, sep=' ', verbose=True):
    """
    Print function integrated with color.
    """
    if verbose:
        color_map = {'r':'red', 'g':'green', 'b':'blue', 'y':'yellow', 'm':'magenta', 'c':'cyan', 'w':'white'}
        if color is None:
            print(*message)
        else: 
            color = color_map[color] if len(color) == 1 else color
            print(colored(sep.join(map(str,message)), color=color, on_color=on_color))

def check_dir(*arg, isFile=False):
    path = os.path.join(*(arg[0:-1])) if isFile else os.path.join(*arg)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return os.path.join(path, arg[-1]) if isFile else path

def get_items_from_file(filelist, useyaml=False, usejson=False, sep='\n'):
    """
    Simple wrapper for reading items from file.
    If file is dumped by yaml or json, set `useyaml` or `userjson` flag ON.
    """
    if not os.path.isfile(filelist):
        raise FileNotFoundError('No such file: {}'.format(filelist))

    with open(filelist, 'r') as f:
        if useyaml:
            lines = yaml.load(f)
        elif usejson:
            lines = json.load(f)
        else:
            lines = f.read().split(sep)
    return lines

def recursive_glob(searchroot='.', searchstr='', verbose=False):
    """
    recursive glob with one search keyword
    """
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    Print("search for {0} in {1}".format(searchstr,searchroot), verbose=verbose)
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if searchstr in filename]
    f.sort()
    return f

def recursive_glob2(searchroot='.', searchstr1='', searchstr2='', verbose=False):
    """
    recursive glob with two search keywords
    """

    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    Print("search for {} and {} in {}".format(searchstr1,searchstr2,searchroot), verbose=verbose)
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if (searchstr1 in filename and searchstr2 in filename)]
    f.sort()
    return f

def recursive_glob3(searchroot='.', searchstr_list=None, excludestr_list=None, verbose=False):
    """
    searchroot: search root dir.
    searchstr_list: search keywords list. [required]
    excludestr_list: not search keywords list. [optional]
    """
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))

    if searchstr_list is None:
        raise ValueError('Search keyword list is empty!')

    Print('search for:', searchstr_list, 'exclude:', excludestr_list ,'in: ', searchroot, verbose=verbose)

    f = []
    for rootdir, dirnames, filenames in os.walk(searchroot):
        for incl in searchstr_list:
            filenames = [fname for fname in filenames if incl in fname]
        for fname in filenames:
            f.append(os.path.join(rootdir, fname))

    if excludestr_list is not None:
        for ex in excludestr_list:
            f = [fname for fname in f if ex not in fname]

    f.sort()
    return f

def PrintProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '=', display=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if display:
        iteration = iteration + 1
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total:
            print()

def save_sourcecode(code_rootdir, out_dir, file_type='*.py', verbose=True):
    if not os.path.isdir(code_rootdir):
        raise FileNotFoundError('Code root dir not exists! {}'.format(code_rootdir))
    Print('Backup source code under root_dir:', code_rootdir, color='red', verbose=verbose)
    outpath = check_dir(out_dir, 'source_code_{}.tar'.format(time.strftime("%m%d_%H%M")), isFile=True)
    tar_opt = 'cvf' if verbose else 'cf'
    os.system("find {} -name '{}' | tar -{} {} -T -".format(code_rootdir, file_type, tar_opt, outpath))

def plot_confusion_matrix(y_true, y_pred,
                          filename, labels,
                          ymap=None, figsize=(10, 10),
                          true_axis_label='Ground-truth',
                          pred_axis_label='Preditions'):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models. with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = true_axis_label
    cm.columns.name = pred_axis_label
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)