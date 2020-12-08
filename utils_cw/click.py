import os, sys, json, subprocess
from termcolor import colored
import click as cli
from pathlib import Path
from .utils import Print

class PathlibEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return os.fspath(obj)
        return json.JSONEncoder.default(self, obj)

def print_smi(ctx, param, value):
    '''Callback for printing nvidia-smi

    If value is set to True, then print the info, vice versa. 
    
    '''
    if value:
        subprocess.call(["nvidia-smi"])

def confirmation(ctx, param, value, output_dir=None, 
                 output_dir_ctx=None, save_code=None, exist_ok=True):
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

    if cli.confirm(colored('Continue processing with these params?\n{}'.format(
            json.dumps(ctx.params, indent=2, sort_keys=True, cls=PathlibEncoder)), color='cyan'), default=True, abort=True):
        
        if output_dir_ctx is not None:
            out_dir = Path(ctx.params[output_dir_ctx])
        elif output_dir is not None:
            out_dir = Path(output_dir)
        else:
            Print('No output dir specified! Do nothing!', color='y')
            return

        out_dir = check_dir(out_dir, exist_ok=exist_ok)
        if out_dir.is_dir():
            with open( out_dir/'param.list','w') as f:
                json.dump(ctx.params, f, indent=2, sort_keys=True, cls=PathlibEncoder)

            file_path = Path(sys.argv[0]).resolve()
            if save_code is None:
                save_code = cli.confirm(colored(f'Save source code of dir:\n{file_path.parent}', color='cyan'), default=True)
            if save_code:
                save_sourcecode(file_path.parent, out_dir)
    

def output_dir_check(ctx, param, value):
    from .utils import check_dir

    if Path.is_dir(value):
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

def prompt_when(ctx, param, value, keyword, trigger=True):
    ''' Only prompt when trigger specified keyword

    # Arguments
        keyword: keyword appeared in ctx.params
        trigger: trigger prompt when keyword == trigger
    '''
    if keyword in ctx.params and ctx.params[keyword] is trigger:
        prompt_string = '\t--> ' + param.name.replace('_', ' ').capitalize()
        Print(f'This option appears because you triggered: {keyword} = {trigger}', color='y')
        return cli.prompt(prompt_string, default=value, type=param.type, \
                          hide_input=param.hide_input, confirmation_prompt=param.confirmation_prompt)
    else:
        return value

def volume_snapshot(data, slice_percentile=50, axis:int=0, output_fname=None, **kwargs):
    '''
    data: input 3d volume, must be normlized
    slice_percentile: (int, tuple) int for single image, tuple for gif slice range
    axis: output axis
    output_fname: output image file name, must include ext
    duration: (optional) set duration time for gif animation
    loop: (optional) set loop time for gif animation
    '''
    from PIL import Image
    import numpy as np
    import collections

    duration = kwargs.get('duration', 40)
    loop     = kwargs.get('loop', 0)
        
    checker = lambda x: min(99, max(0,x))

    slice_num = data.shape[axis]

    if isinstance(slice_percentile, int):
        slice_percentile = checker(slice_percentile)
        slices = [int(slice_num*(slice_percentile/100))]
    elif isinstance(slice_percentile, collections.Sequence):
        slice_percentiles = [checker(slice_percentile[0]),checker(slice_percentile[1])]
        slices = [int(slice_num*(slice_percentiles[0]/100)), int(slice_num*(slice_percentiles[1]/100))]
        slices = np.arange(slices[0], slices[1])


    img_list = []
    for slice_idx in slices:
        slice_data = np.take(data, slice_idx, axis=axis)
        slice_8bit = np.multiply(slice_data, 255)
        pil_img = Image.fromarray(slice_8bit.astype(np.uint8))
        img_list.append(pil_img)

    if '.gif' in output_fname:
        img_list[0].save(output_fname, save_all=True, append_images=img_list[1:], duration=duration, loop=loop)
    else:
        [im.save(output_fname) for i,im in enumerate(img_list)]
    
