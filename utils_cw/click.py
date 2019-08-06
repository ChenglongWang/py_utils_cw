import os, sys, json, subprocess
from termcolor import colored
import click as cli

def print_smi(ctx, param, value):
    '''Callback for printing nvidia-smi

    If value is set to True, then print the info, vice versa. 
    
    '''
    if value: 
        subprocess.call(["nvidia-smi"])

def confirmation(ctx, param, value, output_dir=None, output_dir_ctx=None, save_code=None):
    '''Callback for confirmation
       
    You can use functools.partial() to modify the params
    e.g. functools.partial(confirmation, output_dir='./')

    # Arguments
        output_dir: output dir for save params and source code. 
                    Skip saving if output dir does not exist.
        output_dir_ctx: use this only if you wan use the param specified in previous cmd line.
                        The priority of `output_dir_ctx` is higher than `output_dir`.
                        `output_dir_ctx`='out_dir' will use the ctx.params['out_dir']
        save_code: Set to True/False to enable/disable saving code.
                   Default is None, will request for confirmation every time. 
    '''
    from .utils import save_sourcecode, check_dir
    
    if cli.confirm(colored('Continue processing with these params?\n{}'.format(json.dumps(ctx.params,indent=2,sort_keys=True)), color='cyan'), default=True, abort=True):
        try:
            out_dir = ctx.params[output_dir_ctx]
        except:
            out_dir = output_dir
        
        out_dir = check_dir(out_dir)
        if out_dir and os.path.isdir(out_dir):
            with open( os.path.join(out_dir,'param.list'),'w') as f:
                json.dump(ctx.params, f, indent=2, sort_keys=True)

            file_path = os.path.abspath(sys.argv[0])
            if save_code is None:
                save_code = cli.confirm(colored('Save source code of dir:\n{}'.format(os.path.dirname(file_path)), color='cyan'), default=True)
            if save_code:
                save_sourcecode(os.path.dirname(file_path), out_dir)

def output_dir_check(ctx, param, value):
    from .utils import check_dir
    
    if os.path.isdir(value):
        return value
    else:
        if cli.confirm('Output dir not exists! Do you want to create new one?', default=True, abort=True):
            return check_dir(value)

def output_dir_name(ctx, param, value, parent_dir=None):
    from .utils import check_dir

    dir_path = os.path.join(parent_dir, value) if parent_dir else value

    if os.path.isdir(dir_path):
        return dir_path
    elif cli.confirm('Output dir not exists! Do you want to create new one?\n{}'.format(dir_path), default=True, abort=True):
        return check_dir(dir_path)

def prompt_when(ctx, param, value, trigger):
    from .utils import Print
    if trigger in ctx.params and ctx.params[trigger]:
        prompt_string = '\t--> ' + param.name.replace('_', ' ').capitalize()
        Print('This option appears because you triggered:', trigger, color='y')
        return cli.prompt(prompt_string, default=value, type=param.type, \
                          hide_input=param.hide_input, confirmation_prompt=param.confirmation_prompt)
    else:
        return value

def prompt_when_not(ctx, param, value, trigger):
    from .utils import Print
    if trigger in ctx.params and not ctx.params[trigger]:
        prompt_string = '\t--> ' + param.name.replace('_', ' ').capitalize()
        Print('This option appears because you did not trigger:', trigger, color='y')
        return cli.prompt(prompt_string, default=value, type=param.type, \
                          hide_input=param.hide_input, confirmation_prompt=param.confirmation_prompt)
    else:
        return value