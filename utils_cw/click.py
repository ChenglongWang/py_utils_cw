import os
import sys
import json
import subprocess
from termcolor import colored
import click as cli

def print_smi(ctx, param, value):
    '''Callback for printing nvidia-smi

    If value is set to True, then print the info, vice versa. 
    
    '''
    if value: 
        subprocess.call(["nvidia-smi"])

def confirmation(ctx, param, value, output_dir='./output_dir', save_code=None):
    '''Callback for confirmation
       
    You can use functools.partial() to modify the params
    e.g. functools.partial(confirmation, output_dir='./')

    # Arguments
        output_dir: output dir for save params and source code. 
                    Skip saving if output dir does not exist.
        save_code: Set to True/False to enable/disable saving code.
                   Default is None, will request for confirmation every time. 
    '''
    from .utils import save_sourcecode
    
    if cli.confirm(colored('Continue processing with these params?\n{}\n'.format(json.dumps(ctx.params,indent=2)), color='cyan'), default=True, abort=True):
        if os.path.isdir(output_dir):
            with open( os.path.join(output_dir,'param.list'),'w') as f:
                json.dump(ctx.params, f, indent=2)

            if save_code is None:
                save_code = cli.confirm(colored('Save source code?\n{}'.format(os.path.dirname(sys.argv[0])), color='cyan'), default=True)
            if save_code:
                save_sourcecode(os.path.dirname(sys.argv[0]), output_dir) #better use os.path.abspath(sys.argv[0])?

def output_dir_check(ctx, param, value):
    from .utils import check_dir
    
    if os.path.isdir(value):
        return value
    else:
        if cli.confirm('Output dir not exists! Do you want to create new one?', default=True, abort=True):
            return check_dir(value)
