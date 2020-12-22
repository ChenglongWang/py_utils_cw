# -*- coding: utf-8 -*-

# Abdomen
in_image_search = '.nii'
in_label_search = '.gz'
out_label_search = 'vessel'

default_win_min=-1500   
default_win_max=3000 # basically full HU range (+2000 offeset in moriRaw)
default_img_dir = r'/homes/cwang/Workdata/bronches/CT'
default_label_dir = r'/homes/cwang/Workdata/bronches/GT'
default_out_root = r'/homes/cwang/Workdata/bronches/CT_preproceed'
default_mask_dir = r'/homes/cwang/Workdata/bronches/Lung'
list_file   = r'/suedata1/Free/cwang/WorkData/IJCARS/groundtruth/3_Labels/datalist.txt'
FLIP_DATA=None
USE_LABELS = None
#USE_LABELS = [7] #chiba

##########################################################
ZERO_MEAN=False
NORM=True
IGNORE_VALUE=255

USE_BODY=True
RESAMPLE_DATA = False#[0.7,0.7,0.7]; CORRECT_CPANCREAS_ORIENT=True # [0.6718, 0.6718, 0.501327]
IGNORE_GT = False
## Visceral on Torso  (ACC online network, Stage 1) #######################################################
ZERO_MEAN=False
NORM=True
DILATE_MASK_TO_INCLUDE = 0 # 
RESAMPLE_MASK = False
CROP = False
SWAP_LABELS = None
dx = 2
dy = 2
dz = 2
EXTRACT_FEATURES = False

##########
crop_marginx = 0
crop_marginy = 0
crop_marginz = 0

rBody = 2

######################## FUNCTIONS ###############################
import numpy as np
import os, h5py, sys, mori, click, functools, json
import nibabel as nib
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scipy import ndimage as ndi
from skimage import morphology, measure
from skimage.transform import resize
from types import SimpleNamespace as sn
from utils_cw import recursive_glob, recursive_glob2, Print, check_dir, \
                     Normalize, Normalize2, confirmation, prompt_when

def read_image_info(filename):
    basename = os.path.basename(filename)
    if '.nii' in basename:
        img = nib.load(filename)
        size = img.shape
        spacing = img.affine.diagonal()[0:3]
    elif '.header' in basename:
        hdr = mori.read_mori_header(filename)
        size = hdr['size']
        spacing = hdr['spacing']
    else:
        raise TypeError('Only nifti and mori header files supported! Not {}'.format(filename))
    return size,spacing

def read_image(filename,dtype=None):
    basename = os.path.basename(filename)
    if '.nii' in basename:
        img = nib.load(filename)
        spacing = img.header['pixdim'][1:4]
        print('nifti:',img.shape,img.get_data_dtype(),filename)
        I = img.get_data()
    else: 
        I, hdr = mori.read_mori(filename,dtype)
        if hdr is not None:
            spacing = hdr['spacing']
        else:
            spacing = [1, 1, 1]
    print('{}: {}, spacing {}'.format(basename,np.shape(I),spacing))
    return I, spacing

def resize_segmentation(segmentation, new_shape, order=1, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

############ Functions ###################
def convert_image_and_label_to_h5(image_file,label_file,out_file,\
                                  mask_file=None,DILATE_MASK_TO_INCLUDE=0,\
                                  window_range=(1200,5000),use_body_mask=True,\
                                  use_bin_label=False,reso=(0,0,0), norm_type=None,\
                                  downsample=0,save_nii=True,\
                                  offset=0,zero_mean=False,zero_one_mean=True,\
                                  crop_use_gt=False, min_z_dim=48):
    if not os.path.isfile(image_file):
        raise ValueError('image file does not exist: {}'.format(image_file))
    if label_file is not None and not os.path.isfile(label_file):
        raise ValueError('label file does not exist: {}'.format(label_file))    

    #print('image: {}\nlabel: {}\nout: {}\nmask: {}'.format(image_file,label_file,out_file,mask_file))
    outdir = os.path.split(out_file)[0]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    win_min, win_max = window_range
    
    dx = dy = dz = downsample

    I, i_spacing = read_image(image_file,dtype='>u2') # dtype is only used with raw     
    I = np.asarray(I,dtype=np.float32)
    i_spacing = i_spacing.tolist()

    if offset != 0:
        I = I + offset
    
    print('image spacing{}'.format(i_spacing))
    
    if label_file is not None:    
        L, l_spacing = read_image(label_file,dtype='>u1')
        #tmp for hinyouki
        #L[L==2] = 1
        #L[L==3] = 2 
        if use_bin_label:
            L[L>0] = 1
    else:
        L = np.zeros(np.shape(I),dtype=np.uint8)
        l_spacing = i_spacing
        
    print('label spacing {}'.format(l_spacing))
        
    if FLIP_DATA is 'NIH':
        L = L[::-1,::,::-1]
        I = I[::-1,::,::-1]
    elif FLIP_DATA is 'Visceral':
        L = L[::,::-1,::]
        I = I[::,::-1,::]
    elif FLIP_DATA is 'CTCompAnaB':
        I = I[::-1,::-1,::]
    elif FLIP_DATA is 'Urolo':
        I = I[::,::,::-1]
        L = L[::,::,::-1]
    else:
        print('No flipping')
        
    if reso != [0,0,0]:
        size0 = I.shape
        spacing0 = np.abs(i_spacing)
        if -1 in reso:
            i_spacing = np.multiply([i_spacing[1], i_spacing[1], i_spacing[1]], downsample)
        else:
            i_spacing = np.multiply(reso, downsample)
        
        print('spacing0 =', spacing0, '=>', i_spacing)
        sizeI = np.round(np.divide(np.multiply(size0,spacing0),i_spacing)).astype(np.int)
        xi = np.linspace(0,size0[0]-1,sizeI[0])
        yi = np.linspace(0,size0[1]-1,sizeI[1])
        zi = np.linspace(0,size0[2]-1,sizeI[2])
        XI, YI, ZI = np.meshgrid(xi, yi, zi)            
        # I = interp3([0,size0[0]],[0,size0[1]],[0,size0[2]],I, XI, YI, ZI, method="linear")
        # I = np.transpose(I, [1,0,2])
        I = resize(I, sizeI, order=3, cval=0, mode='edge', anti_aliasing=False)
        L = resize_segmentation(L, sizeI, order=1, cval=0)
        print('Resize IMAGE shape {} to Image shape {}'.format(size0,I.shape))
                       
    if np.any((np.asarray(I.shape)-np.asarray(L.shape))!=0):
        #raise ValueError('image and label have different dimensions!')
        Print('[WARNING] image and label have different dimensions! {}!={} => skip case'.format(I.shape, L.shape), color='r')
        return
    
    # i_spacing_ = i_spacing
    # if downsample > 1:
    #     L = L[::dx,::dy,::dz]
    #     I = I[::dx,::dy,::dz]
    #     print(' downsampled with ({},{},{}) to {}'.format(dx,dy,dz,np.shape(L)))        
    #     i_spacing_[0] = i_spacing[0] * dx 
    #     i_spacing_[1] = i_spacing[1] * dy 
    #     i_spacing_[2] = i_spacing[2] * dz
    #     print('Spacing after downsampling:[{},{},{}]'.format(i_spacing_[0], i_spacing_[1], i_spacing_[2]))

    if USE_LABELS is not None:
        Ltmp = np.copy(L)
        L[...] = 0
        for use_idx,use_label in enumerate(USE_LABELS):
            print('USE_LABEL: map {} to {}'.format(use_label,use_idx+1))
            L[Ltmp==use_label] = use_idx+1

    s_spacing = np.eye(4)
    s_spacing[0,0] = i_spacing[0]
    s_spacing[1,1] = i_spacing[1]
    s_spacing[2,2] = i_spacing[2]
    
    # only learn under body mask 
    if use_body_mask:
        BODY = extract_body_mask_holger(I)
    else:
        BODY = np.zeros(I.shape,dtype=np.bool)
        print('USE ALL VOXELS...')

    if mask_file is not None:
        print('load mask from {}'.format(mask_file))
        MASK, m_spacing = read_image(mask_file,dtype='>u1')
             
        if RESAMPLE_MASK:
            xi = np.linspace(0,MASK.shape[0]-1,I.shape[0])
            yi = np.linspace(0,MASK.shape[1]-1,I.shape[1])
            zi = np.linspace(0,MASK.shape[2]-1,I.shape[2])
            XI, YI, ZI = np.meshgrid(xi, yi, zi)            
            print('Interp3 of MASK shape {} to Image shape {}'.format(MASK.shape,I.shape))
            MASK = interp3([0,MASK.shape[0]],[0,MASK.shape[1]],[0,MASK.shape[2]],MASK,\
                           XI, YI, ZI,\
                           method="nearest")        
            if np.any(MASK.shape!=I.shape):                           
                raise ValueError('Upsampling mask did not work! MASK shape {} to Image shape {}'.format(MASK.shape,I.shape))
            nib.save( nib.Nifti1Image(np.asarray(MASK,dtype=np.uint8),s_spacing), out_file.replace('.h5','--mask-interp.nii.gz') )                
    
        #if RESAMPLE_MASK and DOWNSAMPLE:
        if downsample > 1:            
            MASK = MASK[::dx,::dy,::dz]
            print(' downsampled mask with ({},{},{}) to {}'.format(dx,dy,dz,np.shape(MASK)))              
        MASK = MASK>0  # use all foreground  
    else:             
        MASK = np.zeros(I.shape,dtype=np.bool)
        print('USE NO MASK...')
        
    if DILATE_MASK_TO_INCLUDE>0:
        struct = np.ones((3,3,3),dtype=np.bool)
        print('Dilate MASK>0 with {} iterations...'.format(DILATE_MASK_TO_INCLUDE))
        MASK = ndi.binary_dilation(MASK>0,structure=struct,iterations=DILATE_MASK_TO_INCLUDE) > 0        

    MASK = MASK | BODY    
    MASK0 = np.copy(MASK) # This is saved as *--mask.nii.gz for later candidate generation
    MASK[L>0] = True # make sure labels are within mask!        
                
    # crop based on largest connected component in mask    
    if CROP:
        MASK_labels = measure.label(np.asarray(MASK, dtype=np.int))
        props = measure.regionprops(MASK_labels)
        areas = []
        for prop in props: 
            areas.append(prop.area)
        # only keep largest          
        MASK = MASK_labels==(np.argmax(areas)+1)
        xyz = np.asarray(np.where(MASK>0),dtype=np.int)
        print('Cropping based on indices {}'.format(np.shape(xyz)))
        minx = np.min(xyz[0,::])
        maxx = np.max(xyz[0,::])
        miny = np.min(xyz[1,::])
        maxy = np.max(xyz[1,::])
        minz = np.min(xyz[2,::])
        maxz = np.max(xyz[2,::])        
        print('  found ranges x: {} to {}'.format(minx,maxx))
        print('               y: {} to {}'.format(miny,maxy))
        print('               z: {} to {}'.format(minz,maxz))
        L = L[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        I = I[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        MASK = MASK[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        MASK0 = MASK0[minx:maxx+1,miny:maxy+1,minz:maxz+1]
        print(' cropped to {}'.format(np.shape(L)))        
        with open(out_file.replace('.h5','--crop.txt'), 'w') as f:
            f.write('dim, min, max\n')
            f.write('x, {}, {}\n'.format(minx,maxx))
            f.write('y, {}, {}\n'.format(miny,maxy))
            f.write('z, {}, {}\n'.format(minz,maxz))        
    
    if crop_use_gt:
        margin = 10
        r = np.any(L, axis=(1, 2))
        rmin, rmax = np.where(r)[0][[0, -1]]
        rmin = max(0, rmin-margin)
        rmax = min(L.shape[0], rmax+margin)
        while rmax-rmin < min_z_dim:
            rmax = min(L.shape[0], rmax+1)
            rmin = max(0, rmin-1)
        L = L[rmin:rmax,:,:]
        I = I[rmin:rmax,:,:]
        MASK = MASK[rmin:rmax,:,:]
        MASK0 = MASK0[rmin:rmax,:,:]

    Nvalid = np.sum(MASK)
    Nvoxels = np.size(MASK)
    print('Use {} of {} voxels within mask ({} %)'.format(Nvalid,Nvoxels,100*float(Nvalid)/Nvoxels))
    assert(Nvalid>0)
        
    # correct label image    
    L = np.asarray(L, np.uint8) # use anything larger 0
    if SWAP_LABELS is not None:
        if len(SWAP_LABELS) != 2:
            raise ValueError('SWAP_LABELS only supports 2 labels!')
        xyz0 = np.asarray(np.nonzero(L==SWAP_LABELS[0])).T
        xyz1 = np.asarray(np.nonzero(L==SWAP_LABELS[1])).T
        if np.ptp(xyz1) > np.ptp(xyz0): # assume atery should larger extent (in all directions)
            Ltmp = np.copy(L)
            f = open(out_file.replace('.h5','--swapped.log'), 'w')
            f.close()
            print('swap {}...'.format(SWAP_LABELS))
            L[Ltmp==SWAP_LABELS[0]] = SWAP_LABELS[1]    
            L[Ltmp==SWAP_LABELS[1]] = SWAP_LABELS[0]    
        else:
            print('do not swap labels...')
    
    #L[~MASK] = IGNORE_VALUE
    l, lc = np.unique(L,return_counts=True)
    lc = lc[l!=IGNORE_VALUE] 
    l = l[l!=IGNORE_VALUE]
    print('Labels:')
    frac = []
    for cidx, c in enumerate(lc):
        print(cidx)
        frac.append(float(c)/Nvalid)  

    # compute weights that sum up to 1        
    # generate balanced weight
    weights = np.ndarray(np.shape(I),dtype=np.float32)
    weights.fill(0.0)
    w = []
    if len(lc)>1:
        for cidx, c in enumerate(lc):        
            wc = (1.0-frac[cidx])/(len(lc)-1) # 
            w.append(wc)
            print('  {}: {} of {} ({} percent, w={})'.format(l[cidx],c,np.size(L),100*float(c)/np.size(L),wc))
            weights[L==l[cidx]] = wc
    else:
        print('[WARNING] all voxels have the same label: {}'.format(lc))
        w.append(1.0)
        weights[...] = 1.0
    print('sum(w) = {}'.format(np.sum(w)))        
    if np.abs(1.0-np.sum(w)) > 1e-8:
        print('sum(w) != 1.0, but {}'.format(np.sum(w)))
    weights[~MASK] = 0.0 # ignore in cost function but also via label IGNORE_VALUE
    
    # image windowing
    I[I<win_min] = win_min
    I[I>win_max] = win_max
    if norm_type == 'min-max':
        I = Normalize2(I, True)
    elif norm_type == 'z-score':
        I = Normalize(I, True)
    elif norm_type == 'fixed':
        I = (I-win_min) / (win_max-win_min)
    
    Print('min/max: {}/{}, mean {}'.format(np.min(I),np.max(I),np.mean(I)), color='g')
    
    if zero_one_mean:
        I = Normalize2(I, False)
        Print('Zero2One MEAN: {}, min/max: {}/{}'.format(np.mean(I),np.min(I),np.max(I)), color='g')

    if zero_mean:
        I = I - np.mean(I[MASK])
        Print('ZERO MEAN: {},min/max normed: {}/{}'.format(np.mean(I),np.min(I),np.max(I)), color='g')
    
    if np.any(np.asarray(np.shape(I))-np.asarray(np.shape(L))):
        raise ValueError('image and label have different sizes!')
    

    if save_nii:
        print('save nifti images.')
        #s_spacing = np.eye(4) if np.any(np.less(s_spacing, 0)) else s_spacing
        nib.save( nib.Nifti1Image(I,s_spacing), out_file.replace('.h5','--data.nii.gz') )
        nib.save( nib.Nifti1Image(L,s_spacing), out_file.replace('.h5','--label.nii.gz') )
        nib.save( nib.Nifti1Image(weights,s_spacing), out_file.replace('.h5','--weights.nii.gz') )
        nib.save( nib.Nifti1Image(np.asarray(MASK0,dtype=np.uint8),s_spacing), out_file.replace('.h5','--mask.nii.gz') )    
    
    Print('save h5 as {}...'.format(out_file), color='g')
    with h5py.File(out_file,'w') as h5f:    
        h5f.create_dataset('data',data=I[np.newaxis,np.newaxis,:,:,:],dtype=np.float32,compression='gzip', compression_opts=5) # int16
        h5f.create_dataset('label',data=L[np.newaxis,np.newaxis,:,:,:],dtype=np.uint8,compression='gzip', compression_opts=5)
        h5f.create_dataset('mask', data=MASK0[np.newaxis,np.newaxis,:,:,:],dtype=np.uint8,compression='gzip', compression_opts=5)
        h5f.create_dataset('weights',data=weights[np.newaxis,np.newaxis,:,:,:],dtype=np.float32,compression='gzip', compression_opts=5)
        
        print('saved data ',np.shape(h5f.get('data')))
        print('saved label ',np.shape(h5f.get('label')))
        print('saved mask  ',np.shape(h5f.get('mask')))
        print('saved weights ',np.shape(h5f.get('weights')))
                 
    print('...done.')

def extract_body_mask_holger(I):
    Print('USE BODY MASK Holger ver.', color='y')
    mean_value = np.mean(I)
    BODY = (I>=mean_value)# & (I<=win_max)
    print(' {} of {} voxels masked.'.format(np.sum(BODY),np.size(BODY)))
    if np.sum(BODY)==0:
        raise ValueError('BODY could not be extracted!')
    # Find largest connected component in 3D
    struct = np.ones((3,3,3),dtype=np.bool)
    BODY = ndi.binary_erosion(BODY,structure=struct,iterations=rBody)
    if np.sum(BODY)==0:
        raise ValueError('BODY mask disappeared after erosion!')        
    BODY_labels = measure.label(np.asarray(BODY, dtype=np.int))
    props = measure.regionprops(BODY_labels)
    areas = []
    for prop in props: 
        areas.append(prop.area)
    print('  -> {} areas found.'.format(len(areas)))
    # only keep largest, dilate again and fill holes                
    BODY = ndi.binary_dilation(BODY_labels==(np.argmax(areas)+1),structure=struct,iterations=rBody)
    # Fill holes slice-wise
    BODY = ndi.binary_fill_holes(BODY)
    #for z in range(0,BODY.shape[2]):    
    #    BODY[:,:,z] = ndi.binary_fill_holes(BODY[:,:,z])
    
    return BODY
        
def interp3(xrange, yrange, zrange, v, xi, yi, zi, **kwargs):
    #http://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy
    #from numpy import array
    from scipy.interpolate import RegularGridInterpolator as rgi

    x = np.arange(xrange[0],xrange[1])
    y = np.arange(yrange[0],yrange[1])
    z = np.arange(zrange[0],zrange[1])
    interpolator = rgi((x,y,z), v, **kwargs)
    
    pts = np.array([np.reshape(xi,(-1)), np.reshape(yi,(-1)), np.reshape(zi,(-1))]).T    
    Vi = interpolator(pts)
    return np.reshape(Vi, np.shape(xi))        
          
@click.group()
def main():
    pass

def split_multi_input(ctx, param, value):
    if value.lower() == 'all':
        return 'all'
    
    split_char = ',' if ',' in value else ' '
    return [ float(v) for v in value.split(split_char)]

@main.command('airway')
@click.option('--img-dir', prompt=True, type=click.Path(True), default=default_img_dir, help='Image dir')
@click.option('--single-file', type=str, default='', help='For processing single file')
@click.option('--label-dir', prompt=True, type=click.Path(True), default=default_label_dir, help='Label dir')
@click.option('--mask-dir', prompt=True, type=click.Path(True), default=default_mask_dir, help='Mask dir')
@click.option('--out-dir', prompt=True, type=str, default=default_out_root, help='Output dir')
@click.option('--keyword', prompt=True, type=str, default='.nii.gz', help='Keyword for searching image')
@click.option('--use-body', prompt=True, type=bool, default=USE_BODY, help='Use body mask.')
@click.option('--offset', prompt=True, type=int, default=0, help='Offset for CT value. (Do not change if u know what is this!)')
@click.option('--window', type=(int,int), default=(default_win_min,default_win_max), help='Window range for intensity clipping [1200,5000]')
@click.option('--reso', type=(float,float,float), default=(0,0,0), help='Resample data. (0,0,0)->disable;(-1,-1,-1)->use xy reso;(a,b,c)->customize' )
@click.option('--binary-label', type=bool, default=True, help='Force to use binary label')
@click.option('--save-nii', is_flag=True)
def convert(**args):
    sargs = sn(**args)

    images_with_no_labels, images_with_no_masks = [], []
    not_found_count = 0    
    if sargs.single_file:
        image_files = recursive_glob2(sargs.img_dir, sargs.single_file, sargs.keyword)
    else:
        image_files = recursive_glob(sargs.img_dir, sargs.keyword)

    if len(image_files) == 0:
        Print('No data found!', color='r')
        return

    N = len(image_files)    
    Nstep = np.ceil(float(N))
    start_from = 0
    
    for idx in range(start_from,N):
        #image = recursive_glob2(sargs.img_dir, images[idx].strip('gz'), in_image_search, 'dummy')
        image = image_files[idx]
          
        basename = os.path.basename(image).strip(sargs.keyword)
        
        label = recursive_glob2(sargs.label_dir,basename,sargs.keyword) if sargs.label_dir is not None else None
        
        if not label:
            Print('[WARNING] No unique label found for {}'.format(image), color='y')
            images_with_no_labels.append(image)
        else:
            label = label[0]

        mask = recursive_glob2(sargs.mask_dir,basename,sargs.keyword) if sargs.mask_dir is not None else None

        if not mask:
            Print('[WARNING] No unique mask found for {}'.format(image), color='y')
            images_with_no_masks.append(mask)
        else:
            mask = mask[0]

        Print('image:', image, color='g')
        Print('label:', label, color='g')
        Print('mask: ', mask,  color='g')
        
        output_root = check_dir(sargs.out_dir)
        if os.path.exists(os.path.join(output_root, basename)):
            Print('Results already exists! Skip', color='y')
            continue
        output_dir = check_dir(output_root,basename)
        
        h5file1 = os.path.join(output_dir, basename+'.h5')
        convert_image_and_label_to_h5(image,label,h5file1,mask,0,\
                                      sargs.window,sargs.use_body,\
                                      sargs.binary_label,sargs.reso,\
                                      sargs.save_nii,sargs.offset,ZERO_MEAN,NORM)
        
    print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
    for i in images_with_no_labels: print(i)
    print('Not found {} of {} images'.format(not_found_count,N))    

default_win_min=-1024   
default_win_max=1700
kits_img_dir = r'/homes/cwang/kits19/data/imagesTr'
kits_label_dir = r'/homes/cwang/kits19/data/imagesTr'
kits_out_root = r'/homes/cwang/kits19/norm_data'

@main.command('kits')
@click.option('--img-dir', prompt=True, type=str, default=kits_img_dir, help='Image dir')
@click.option('--file-list', type=str, default='', help='filelist for data')
@click.option('--out-dir', prompt=True, type=str, default=kits_out_root, help='Output dir')
@click.option('--keyword', prompt=True, type=str, default='.nii.gz', help='Keyword for searching image')
@click.option('--use-body', prompt=True, type=bool, default=USE_BODY, help='Use body mask.')
@click.option('--offset', prompt=True, type=int, default=0, help='Offset for CT value. (Do not change if u know what is this!)')
@click.option('--window', type=(int,int), default=(default_win_min,default_win_max), help='Window range for intensity clipping [1200,5000]')
@click.option('--reso', prompt=True, callback=split_multi_input, default="0,0,0", help='Resample data. (0,0,0)->disable;(-1,-1,-1)->use xy reso;(a,b,c)->customize')
@click.option('--downsample', type=int, default=1, help='Downsample ratio. Disable: 0 or 1')
@click.option('--norm-type', prompt=True, type=click.Choice(['min-max', 'z-score', 'fixed'], show_index=True), default=1, help='Choose norm type' )
@click.option('--use-bb', is_flag=True, help='Crop image use groundtruth as boundingbox')
@click.option('--min-z-dim', type=int, default=48, callback=functools.partial(prompt_when,trigger='use_bb'),help='Min z dim size')
@click.option('--save-nii-num', type=int, default=0, help='Save nii examples for checking')
@click.option('--zero-mean', is_flag=True, help='Zero mean flag')
@click.option('--zero2one', is_flag=True, help='0-1 mean flag')
@click.option('--confirm', callback=functools.partial(confirmation, output_dir_ctx='out_dir', save_code=False))
def convert1(**args):
    sargs = sn(**args)
    
    not_found_count = 0
    images_with_no_labels, images_with_no_masks = [], []
    if os.path.isfile(sargs.file_list):
        with open(sargs.file_list, 'r') as f:
            image_files = json.load(f)
    else:
        image_files = recursive_glob2(sargs.img_dir, sargs.keyword, 'imaging')

    if len(image_files) == 0:
        Print('No data found!', color='r')
        return

    N = len(image_files)    
    Nstep = np.ceil(float(N))
    start_from = 0
    
    for idx in range(start_from,N):
        image = image_files[idx]
        label = image.replace('imaging', 'segmentation')
        #label = recursive_glob2(sargs.label_dir,basename,sargs.keyword) if sargs.label_dir is not None else None
        
        if not os.path.isfile(label):
            Print('[ERROR] No label found for {}'.format(label), color='r')
            label = None

        Print('image:', image, color='g')
        Print('label:', label, color='g')
        
        casename = os.path.basename(os.path.dirname(image))
        output_root = check_dir(sargs.out_dir)
        if os.path.exists(os.path.join(output_root, casename)):
            Print('Results already exists! Skip', color='y')
            continue
        output_dir = check_dir(output_root,casename)
        
        h5file1 = os.path.join(output_dir, casename+'.h5')
        convert_image_and_label_to_h5(image, label, h5file1, None, 0,\
                                      sargs.window, sargs.use_body,\
                                      False, sargs.reso, sargs.norm_type,\
                                      sargs.downsample, sargs.save_nii, \
                                      sargs.offset, sargs.zero_mean, sargs.zero2one,\
                                      sargs.use_bb, sargs.min_z_dim )
        
    print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
    for i in images_with_no_labels: print(i)
    print('Not found {} of {} images'.format(not_found_count,N))

    #generate filist
    image_files = recursive_glob(output_root, '.h5')
    with open(os.path.join(output_root, 'filelist'), 'w') as f:
        json.dump(image_files, f, indent=2)


default_win_min=1024
default_win_max=3800
kits_img_dir = r'/data4/CTUrolo/hinyouki_nu_sasa/NiftiFormat'
kits_label_dir = r'/homes/cwang/Workdata/hinyouki-cancer/manual'
kits_out_root = r'/homes/cwang/Workdata/hinyouki-cancer/prepared-data'

@main.command('hinyouki')
@click.option('--img-dir', prompt=True, type=click.Path(True), default=kits_img_dir, help='Image dir')
@click.option('--out-dir', prompt=True, type=str, default=kits_out_root, help='Output dir')
@click.option('--keyword', prompt=True, type=str, default='.nii.gz', help='Keyword for searching image')
@click.option('--use-body', prompt=True, type=bool, default=USE_BODY, help='Use body mask.')
@click.option('--offset', prompt=True, type=int, default=0, help='Offset for CT value. (Do not change if u know what is this!)')
@click.option('--window', type=(int,int), default=(default_win_min,default_win_max), help='Window range for intensity clipping [1200,5000]')
@click.option('--reso', prompt=True, callback=split_multi_input, default="0,0,0", help='Resample data. (0,0,0)->disable;(-1,-1,-1)->use xy reso;(a,b,c)->customize')
@click.option('--downsample', type=int, default=0, help='Downsample ratio. Disable: 0 or 1')
@click.option('--norm-type', prompt=True, type=click.Choice(['min-max', 'z-score', 'fixed'], show_index=True), default=1, help='Choose norm type' )
@click.option('--zero-mean', is_flag=True, help='Zero mean flag')
@click.option('--zero2one', is_flag=True, help='0-1 mean flag')
@click.option('--save-nii', is_flag=True)
@click.option('--confirm', callback=functools.partial(confirmation, output_dir_ctx='out_dir', save_code=False))
def convert2(**args):
    sargs = sn(**args)

    images_with_no_labels, images_with_no_masks = [], []
    not_found_count = 0
    label_files = recursive_glob(kits_label_dir, sargs.keyword)

    if len(label_files) == 0:
        Print('No data found!', color='r')
        return

    N = len(label_files)    
    Nstep = np.ceil(float(N))
    start_from = 0
    
    for idx in range(start_from,N):
        label = label_files[idx]
        basename = os.path.basename(label).split('.')[0]
          
        images = recursive_glob2(sargs.img_dir,basename,sargs.keyword,verbose=True)
        
        if len(images) != 1:
            Print('[ERROR] No image found for {}, Skip!'.format(images), color='r')
            continue
        else:
            image = images[0]

        Print('image:', image, color='g')
        Print('label:', label, color='g')
        
        output_root = check_dir(sargs.out_dir)
        if os.path.exists(os.path.join(output_root, basename)):
            Print('Results already exists! Skip', color='y')
            continue
        output_dir = check_dir(output_root,basename)
        
        h5file1 = os.path.join(output_dir, basename+'.h5')
        convert_image_and_label_to_h5(image, label, h5file1, None, 0,\
                                      sargs.window, sargs.use_body,\
                                      False, sargs.reso, sargs.norm_type,\
                                      sargs.downsample, sargs.save_nii, \
                                      sargs.offset, sargs.zero_mean, sargs.zero2one)
        
    print('{} images_with_no_labels:'.format(len(images_with_no_labels)))    
    for i in images_with_no_labels: print(i)
    print('Not found {} of {} images'.format(not_found_count,N))

    #generate filist
    image_files = recursive_glob(output_root, '.h5')
    with open(os.path.join(output_root, 'filelist'), 'w') as f:
        json.dump(image_files, f, indent=2)

if __name__ == '__main__':
    main() 
    


    