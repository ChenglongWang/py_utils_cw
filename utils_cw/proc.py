import random
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from utils_cw.utils import Print

def Normalize(data, verbose=False):
    '''
    Z-score normalization
    '''
    mean_data, std_data = np.mean(data), np.std(data)
    norm_data = (data-mean_data)/std_data
    Print('Normalize mean:', mean_data,'std:',std_data, verbose=verbose)
    return norm_data

def Normalize2(data, verbose=False):
    '''
    Min-Max normalization
    '''
    minValue, maxValue = np.min(data), np.max(data)
    norm_data = (data-minValue) / (maxValue-minValue)
    Print('Normalize2 min:', minValue,'max:',maxValue, verbose=verbose)
    return norm_data.astype(np.float32)

def get_non_body_mask(ct_data, win_min=1200, morphology_radius=1):
    """
    Get non-body mask from CT image.
    Just simple invert of body mask.
    """
    body_mask = get_body_mask(ct_data, win_min, morphology_radius)
    non_body_mask = binary_dilation(np.invert(body_mask), structure=np.ones((3,3,3),dtype=np.bool), iterations=morphology_radius)
    return non_body_mask

def get_body_mask(ct_data, win_min=1200, morphology_radius=2, connectivity=3):
    """
    Get body mask from CT image.
    Argument `win_min` should approx to air intensity value.
    """
    body_mask = (ct_data>=win_min)# & (ct_data<=win_max)
    #print(' {} of {} voxels masked.'.format(np.sum(body_mask),np.size(body_mask)))
    if np.sum(body_mask)==0:
        raise ValueError('BODY could not be extracted!')
    
    # Find largest connected component in 3D
    struct = generate_binary_structure(3,connectivity)
    body_mask = binary_erosion(body_mask,structure=struct,iterations=morphology_radius)
    if np.sum(body_mask)==0:
        raise ValueError('BODY mask disappeared after erosion!')

    # Get the largest connected component
    labeled_array, num_features = label(body_mask, structure=struct)
    component_sizes = np.bincount(labeled_array.ravel())
    max_label = np.argmax(component_sizes[1:-1])+1

    # only keep largest, dilate again and fill holes                
    body_mask = binary_dilation(labeled_array==max_label,structure=struct,iterations=morphology_radius)
    # Fill holes slice-wise
    for z in range(0,body_mask.shape[2]):    
        body_mask[:,:,z] = binary_fill_holes(body_mask[:,:,z])
    return body_mask


def __get_random_center(data, label, crop_size):
    dshape = data.shape[-3:]
    pad_size = np.maximum( np.subtract(crop_size, dshape),  (0,)*3 )
    radius_L = np.divide(crop_size, 2).astype(np.int)
    radius_R = np.subtract(crop_size, radius_L)
    pad_radius_L = np.ceil( np.divide(pad_size, 2) ).astype(np.int)
    pad_radius_R = np.subtract(pad_size, pad_radius_L)
    crop_center  = [random.randint(radius_L[i]-pad_radius_L[i], dshape[i]-radius_R[i]+pad_radius_R[i]) for i in range(3)]
    return crop_center

def __get_gt_center(data, label, crop_size):
    dshape = data.shape[-3:]
    pad_size = np.maximum( np.subtract(crop_size, dshape),  (0,)*3 )
    x_radius_L, y_radius_L, z_radius_L = np.divide(crop_size, 2).astype(np.int)
    x_radius_R, y_radius_R, z_radius_R = crop_size[0]-x_radius_L, crop_size[1]-y_radius_L, crop_size[2]-z_radius_L
    pad_x_radius, pad_y_radius, pad_z_radius = np.ceil( np.divide(pad_size, 2) ).astype(np.int)
    
    # limit in valid bbox
    label_ = label[x_radius_L-pad_x_radius : dshape[0]-x_radius_R+pad_x_radius,
                   y_radius_L-pad_y_radius : dshape[1]-y_radius_R+pad_y_radius,
                   z_radius_L-pad_z_radius : dshape[2]-z_radius_R+pad_z_radius]
    foreground_voxels = np.nonzero(label_)
    try:
        ridx = random.randint(0, np.shape(foreground_voxels)[1])
        crop_center = [int(foreground_voxels[-3][ridx]),
                       int(foreground_voxels[-2][ridx]),
                       int(foreground_voxels[-1][ridx])]
        return np.add(crop_center, [x_radius_L-pad_x_radius, y_radius_L-pad_y_radius, z_radius_L-pad_z_radius])
    except: # No nonzero label is found in valid label region
            # change strategy: if cropped volume contains nonzero label, Fine!
        nonzero_voxel = -1
        while nonzero_voxel < 1:
            crop_center = __get_random_center(data, None, crop_size)
            x, y, z = crop_center
            x_lb, x_ub = max(0, x-x_radius_L), min(dshape[0], x+x_radius_R)
            y_lb, y_ub = max(0, y-y_radius_L), min(dshape[1], y+y_radius_R)
            z_lb, z_ub = max(0, z-z_radius_L), min(dshape[2], z+z_radius_R)
            nonzero_voxel = np.count_nonzero( label[x_lb:x_ub, y_lb:y_ub, z_lb:z_ub] )
        return crop_center

def __crop3D_with_pad(data, label, size, center_fn, crop_center=None, **kwargs):
    verbose = kwargs.get('verbose', False)
    bias = kwargs.get('pos_bias', 0)
    center_threshold = kwargs.get('crop_center_threshold', None)

    dshape = data.shape[-3:]
    if crop_center is None:
        crop_center = center_fn(data, label, size)
        if center_threshold is not None:
            while data[..., crop_center[0],crop_center[1],crop_center[2]]<=center_threshold:
                Print('Crop center intensity is lower than', center_threshold, 'Redo!', color='g', verbose=verbose)
                crop_center = center_fn(data, label, size)

    if isinstance(bias, int):
        bias = (bias,)*3
    if np.any(np.greater(bias, 0)):
        pos_bias = (random.randint(-bias[0], bias[0]),
                    random.randint(-bias[1], bias[1]),
                    random.randint(-bias[2], bias[2]))   
        Print('center pt pos bias:', pos_bias, verbose=verbose)
        crop_center = np.add(crop_center,pos_bias)

    x, y, z = crop_center
    x_radius_L, y_radius_L, z_radius_L = np.divide(size, 2).astype(np.int)
    x_radius_R, y_radius_R, z_radius_R = size[0]-x_radius_L, size[1]-y_radius_L, size[2]-z_radius_L
    x_lb, x_ub = max(0, x-x_radius_L), min(dshape[0], x+x_radius_R)
    y_lb, y_ub = max(0, y-y_radius_L), min(dshape[1], y+y_radius_R)
    z_lb, z_ub = max(0, z-z_radius_L), min(dshape[2], z+z_radius_R)

    pad_x_L = x_radius_L-x if x<x_radius_L else 0
    pad_x_R = x_radius_R+x-dshape[0] if x_radius_R+x>dshape[0] else 0
    pad_y_L = y_radius_L-y if y<y_radius_L else 0
    pad_y_R = y_radius_R+y-dshape[1] if y_radius_R+y>dshape[1] else 0
    pad_z_L = z_radius_L-z if z<z_radius_L else 0
    pad_z_R = z_radius_R+z-dshape[2] if z_radius_R+z>dshape[2] else 0
    pads = (pad_x_L,pad_x_R,pad_y_L,pad_y_R,pad_z_L,pad_z_R)

    crop = data[..., x_lb:x_ub, y_lb:y_ub, z_lb:z_ub]
    Print('Pad_crop at {} with size {}, got size {}'.format(crop_center, size, crop.shape), verbose=verbose)
    Print('Padding: ', pads, verbose=verbose)

    if np.any( np.greater(pads, 0) ):
        if len(data.shape) == 3:
            crop_pad = np.pad(crop, ((pad_x_L, pad_x_R), 
                                     (pad_y_L, pad_y_R),
                                     (pad_z_L, pad_z_R)),
                              mode='constant')
        elif len(data.shape) == 4:
            crop_pad = np.pad(crop, ((0,0),
                                     (pad_x_L, pad_x_R), 
                                     (pad_y_L, pad_y_R),
                                     (pad_z_L, pad_z_R)),
                              mode='constant')
        elif len(data.shape) == 5:
            crop_pad = np.pad(crop, ((0,0),
                                     (0,0),
                                     (pad_x_L, pad_x_R), 
                                     (pad_y_L, pad_y_R),
                                     (pad_z_L, pad_z_R)),
                              mode='constant')
        else:
            raise NotImplementedError("data's shape {} len > 5 not supported yet".format(data.shape))
        
        return crop_pad, crop_center
    else:
        return crop, crop_center

def __crop3D_no_pad(data, label, size, center_fn, crop_center=None, **kwargs):
    '''
    Please insure you want no_pad cropping. Otherwise use __crop3D_with_pad() instead!
    '''
    verbose = kwargs.get('verbose', False)
    center_threshold = kwargs.get('crop_center_threshold', None)

    if crop_center is None:
        crop_center = center_fn(data, label, size)
        if center_threshold is not None:
            while data[..., crop_center[0],crop_center[1],crop_center[2]]<=center_threshold:
                Print('Crop center intensity is lower than', center_threshold, 'Redo!', color='g', verbose=verbose)
                crop_center = center_fn(data, label, size)
                
    x, y, z = crop_center
    dshape = data.shape[-3:]

    x_radius, y_radius, z_radius = np.divide(size, 2).astype(np.int)
    x_lb, x_ub = max(0, x-x_radius), min(dshape[0], x+x_radius)
    y_lb, y_ub = max(0, y-y_radius), min(dshape[1], y+y_radius)
    z_lb, z_ub = max(0, z-z_radius), min(dshape[2], z+z_radius)

    crop = data[..., x_lb:x_ub, y_lb:y_ub, z_lb:z_ub]
    Print('Pad_crop at {} with size {}, got size {}'.format(crop_center,size, crop.shape), verbose=verbose)
    
    return crop, crop_center

def crop3D(data, crop_size, label=None, crop_center=None, pad=(0,0,0), **kwargs):
    '''
    3D volume cropping.
    data: shape can be (n,c,w,h,d) or (c,w,h,d) or (w,h,d). anyway last 3 dims should be WHD.
    label: None-> random cropping; else-> random nonzero point as crop_center
    pad: pad crop size at boarder
    kwargs: optional arguments -> verbose; pos_bias; force_no_pad; crop_center_threshold
    '''

    pad_crop_size = np.add(crop_size, np.multiply(pad,2))
    center_pt_func = __get_random_center if label is None else __get_gt_center
    if kwargs.get('force_no_pad', False):
        return __crop3D_no_pad(data, label, pad_crop_size, center_pt_func, crop_center, **kwargs)
    else:
        return __crop3D_with_pad(data, label, pad_crop_size, center_pt_func, crop_center, **kwargs)