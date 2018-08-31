import os
from termcolor import colored

def Print(*message, color=None, on_color=None, sep=' ', verbose=True):
    if verbose:
        if color is None:
            print(*message)
        else:
            print(colored(sep.join(map(str,message)), color=color, on_color=on_color))

def check_dir(*arg, isFile=False):
    path = os.path.join(*(arg[0:-1])) if isFile else os.path.join(*arg)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return os.path.join(path, arg[-1]) if isFile else path

def recursive_glob(searchroot='.', searchstr=''):
    """
    recursive glob with one search keyword
    """
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print("search for {0} in {1}".format(searchstr,searchroot))
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if searchstr in filename]
    f.sort()
    return f

def recursive_glob2(searchroot='.', searchstr1='', searchstr2=''):
    """
    recursive glob with two search keywords
    """
    
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    print("search for {} and {} in {}".format(searchstr1,searchstr2,searchroot))
    f = [os.path.join(rootdir, filename)
        for rootdir, dirnames, filenames in os.walk(searchroot)
        for filename in filenames if (searchstr1 in filename and searchstr2 in filename)]
    f.sort()
    return f

def recursive_glob3(searchroot='.', searchstr_list=None, excludestr_list=None, verbose=True):
    """
    searchroot: search root dir.
    searchstr_list: search keywords list. [required]
    excludestr_list: not search keywords list. [optional]
    """
    if not os.path.isdir(searchroot):
        raise ValueError('No such directory: {}'.format(searchroot))
    
    if searchstr_list is None:
        raise ValueError('Search keyword list is empty!')

    if verbose:
        print('search for:', searchstr_list, 'exclude:', excludestr_list ,'in: ', searchroot)

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