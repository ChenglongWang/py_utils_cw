import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes

def Normalize(data, verbose=False):
    '''
    Z-score normalization
    '''
    mean_data, std_data = np.mean(data), np.std(data)
    if verbose:
        print('Normalize mean:', mean_data,'std:',std_data)
    norm_data = (data-mean_data)/std_data
    return norm_data

def Normalize2(data, verbose=False):
    '''
    Min-Max normalization
    '''
    minValue, maxValue = np.min(data), np.max(data)
    if verbose:
        print('Normalize2 min:', minValue,'max:',maxValue)
    norm_data = (data-minValue) / (maxValue-minValue)
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
