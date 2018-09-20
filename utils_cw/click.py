import os
import sys
import json
import subprocess
from termcolor import colored
import click as cli
from utils import save_sourcecode

def print_smi(ctx, param, value):
    if value: 
        subprocess.call(["nvidia-smi"])

def confirmation(ctx, param, value):
    if cli.confirm(colored('Continue processing with these params?\n{}\n'.format(json.dumps(ctx.params,indent=2)), color='cyan'), default=True, abort=True):
        with open( os.path.join(ctx.params['output_dir'],'param.list'),'w') as f:
            json.dump(ctx.params, f, indent=2)
    if cli.confirm(colored('Save source code?\n{}'.format(os.path.dirname(sys.argv[0])), color='cyan'), default=True): 
        save_sourcecode(os.path.dirname(sys.argv[0]), ctx.params['output_dir']) #better use os.path.abspath(sys.argv[0])?

def output_dir_check(ctx, param, value):
    from .utils import check_dir
    if os.path.isdir(value):
        return value
    else:
        if cli.confirm('Output dir not exists! Do you want to create new one?', default=True, abort=True):
            return check_dir(value)
