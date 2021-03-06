import os, time, yaml, json, h5py
import numpy as np
from pathlib import Path
from termcolor import colored

def Print(*message, color=None, on_color=None, sep=' ', end='\n', verbose=True):
    """
    Print function integrated with color.
    """
    if verbose:
        color_map = {'r':'red', 'g':'green', 'b':'blue', 'y':'yellow', 'm':'magenta', 'c':'cyan', 'w':'white'}
        if color is None:
            print(*message, end=end)
        else: 
            color = color_map[color] if len(color) == 1 else color
            print(colored(sep.join(map(str,message)), color=color, on_color=on_color), end=end)

def check_dir(*arg, isFile=False, exist_ok=True):
    path = Path(*(arg[0:-1])) if isFile else Path(*arg)
    if not path.is_dir():
        os.makedirs(path, exist_ok=exist_ok)
    return path/arg[-1] if isFile else path

def get_items_from_file(filelist, format='auto', sep='\n'):
    """
    Simple wrapper for reading items from file.
    If file is dumped by yaml or json, set `format` to `json`/`yaml`.
    """
    filelist = Path(filelist)
    if not filelist.is_file():
        raise FileNotFoundError(f'No such file: {filelist}')
    
    if format == 'auto':
        if filelist.suffix in ['.json']:
            format = 'json'
        elif filelist.suffix in ['.yaml', '.yml']:
            format = 'yaml'
        else:
            format = None

    with filelist.open() as f:
        if format=='yaml':
            lines = yaml.full_load(f)
        elif format=='json':
            lines = json.load(f)
        else:
            lines = f.read().split(sep)
    return lines

def load_h5(h5_file:str, keywords:list, transpose=None, verbose=False):
    hf = h5py.File(h5_file, 'r')
    #print('List of arrays in this file: \n', hf.keys())
    if len(keywords) == 0:
        Print('Get all items:', hf.keys(), verbose=verbose)
        keywords = list(hf.keys())

    dataset = [ np.copy(hf.get(key)) if key in hf.keys() else None for key in keywords ]
    if verbose:
        [ Print(f'{key} shape: {np.shape(data)}', color='g') if 
          data is not None else Print(f'{key} is None', color='r') for 
          key, data in zip(keywords, dataset) ]

    if transpose:
        dataset = [np.transpose(data, transpose) if data is not None
            else None for data in dataset
        ]
    return dataset

def recursive_glob(searchroot='.', searchstr='', verbose=False):
    """
    recursive glob with one search keyword
    """
    if not os.path.isdir(searchroot):
        raise ValueError(f'No such directory: {searchroot}')
    
    if '*' not in searchstr:
        searchstr = '*'+searchstr+'*'

    Print(f"search for {searchstr} in {searchroot}", verbose=verbose)
    
    f = [Path(rootdir).joinpath(filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if Path(filename).match(searchstr)]
    f.sort()
    return f

def recursive_glob2(searchroot='.', searchstr1='', searchstr2='', logic='and', verbose=False):
    """
    recursive glob with two search keywords
    """
    if not os.path.isdir(searchroot):
        raise ValueError(f'No such directory: {searchroot}')
    if logic == 'and':
        logic_op = np.logical_and
    elif logic == 'or':
        logic_op = np.logical_or

    if '*' not in searchstr1:
        searchstr1 = '*'+searchstr1+'*'
    if '*' not in searchstr2:
        searchstr2 = '*'+searchstr2+'*'

    Print(f"search for {searchstr1} {logic} {searchstr2} in {searchroot}", verbose=verbose)
    
    f = [Path(rootdir).joinpath(filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if logic_op(Path(filename).match(searchstr1), Path(filename).match(searchstr2))]
    f.sort()
    return f

#! Todo: add logic_op option like recursive_glob2
def recursive_glob3(searchroot='.', searchstr_list=None, excludestr_list=None, verbose=False):
    """
    searchroot: search root dir.
    searchstr_list: search keywords list. [required]
    excludestr_list: not search keywords list. [optional]
    """
    if not os.path.isdir(searchroot):
        raise ValueError(f'No such directory: {searchroot}')

    if searchstr_list is None:
        raise ValueError('Search keyword list is empty!')

    Print('search for:', searchstr_list, 'exclude:', excludestr_list ,'in: ', searchroot, verbose=verbose)

    f = []
    for rootdir, dirnames, filenames in os.walk(searchroot):
        for incl in searchstr_list:
            filenames = [fname for fname in filenames if incl in fname.lower()]
        for fname in filenames:
            f.append(os.path.join(rootdir, fname))

    if excludestr_list is not None:
        for ex in excludestr_list:
            f = [fname for fname in f if ex not in fname.lower()]

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
        raise FileNotFoundError(f'Code root dir not exists! {code_rootdir}')
    Print('Backup source code under root_dir:', code_rootdir, color='red', verbose=verbose)
    outpath = check_dir(out_dir, f"source_code_{time.strftime('%m%d_%H%M')}.tar", isFile=True)
    tar_opt = 'cvf' if verbose else 'cf'
    os.system(f"find {code_rootdir} -name '{file_type}' | tar -{tar_opt} {outpath} -T -")

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