import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from PIL import Image
import numbers


def random_num_generator(config, random_state=np.random, cast_type=None):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    elif config[0] == 'choice':
        ret = random_state.choice(config[1], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')

    if cast_type is None:
        return ret 
    else:
        return cast_type(ret)

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def generate_elastic_transform_coordinates(shape, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    if isinstance(alpha, (int, float)):
        alpha = (alpha,)*len(shape)
    if isinstance(sigma, (int, float)):
        sigma = (sigma,)*len(shape)

    n_dim = len(shape)
    offsets = []
    for i in range(n_dim):
        offsets.append(
            gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma[i], mode="constant", cval=0) * alpha[i])
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing='ij')
    indices = [np.reshape(i + j, (-1, 1)) for i, j in zip(offsets, coords)]
    return indices

def elastic_deform_coordinates(coordinates, alpha, sigma):
    if isinstance(alpha, (int, float)):
        alpha = (alpha,)*len(coordinates)
    if isinstance(sigma, (int, float)):
        sigma = (sigma,)*len(coordinates)

    n_dim = len(coordinates)
    offsets = []
    for i in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma[i], mode="constant", cval=0) * alpha[i])
    indices = np.add(offsets, coordinates)
    return indices

def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_label=False):
    if is_label and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape, img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img==c).astype(float), coords, order=order, mode=mode, cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(img.astype(float), coords, order=order, mode=mode, cval=cval).astype(img.dtype)

def create_matrix_rotation_3d(angle, axis, matrix=None):
    if axis == 'x':
        rotation = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)

def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_3d(angle_x, 'x', rot_matrix)
    rot_matrix = create_matrix_rotation_3d(angle_y, 'y', rot_matrix)
    rot_matrix = create_matrix_rotation_3d(angle_z, 'z', rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords

def scale_coords(coords, scale):
    return coords * scale

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
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
            reshaped_multihot = resize((segmentation==c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resize_multichannel_image(image, new_shape, order=3):
    '''
    Resizes image. Resizes each channel in c separately and fuses results back together
    :param image: c x x x y (x z)
    :param new_shape: x x y (x z)
    :param order:
    :return:
    '''
    tpe = image.dtype
    assert len(image.shape) == len(new_shape), "new shape must have same dimensionality as image"
    if len(image.shape) == 3:
        return resize(image.astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
    elif len(image.shape) == 4:
        new_shp = [image.shape[0]] + list(new_shape)
        result = np.zeros(new_shp, dtype=image.dtype)
        for i in range(image.shape[0]):
            result[i] = resize(image[i].astype(float), new_shape, order, "constant", 0, True, anti_aliasing=False)
        return result.astype(tpe)
    else:
        raise NotImplementedError

##############################################################################


def elastic_transform2D(image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result

def elastic_transform3D(image, label=None, alpha=5, sigma=30, spline_order=1, mode='nearest', random_state=None):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    indices = generate_elastic_transform_coordinates(image.shape, alpha, sigma, random_state)
    if label is None:
        return map_coordinates(image, indices, order=spline_order, mode=mode).reshape(image.shape), None
    else:
        img = map_coordinates(image, indices, order=spline_order, mode=mode).reshape(image.shape)
        lab = map_coordinates(label, indices, order=0, mode=mode).reshape(label.shape)
        return img, lab


def augment_gamma(data_sample, gamma=1.2, retain_stats=False):
    epsilon=1e-7

    if retain_stats:
        mn = data_sample.mean()
        sd = data_sample.std()
    
    minm = data_sample.min()
    rnge = data_sample.max() - minm
    data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
    
    if retain_stats:
        data_sample = data_sample - data_sample.mean() + mn
        data_sample = data_sample / (data_sample.std() + 1e-8) * sd

    return data_sample

def rescaling(data, zoom_rate=1., axis=[0,1,2], order=2):
    origin_shape = data.shape
    assert len(axis) <= len(origin_shape)
    zoom_rates = np.ones(len(origin_shape))
    for i in axis:
        zoom_rates[i] = zoom_rate
    
    if zoom_rate == 1:
        return data
    elif zoom_rate > 1:
        zoomed_data = ndimage.zoom(data, zoom_rates, order=order, mode='constant')
        #cropping to original size
        return center_cropping(zoomed_data, origin_shape)
    else:
        raise NotImplementedError

def augment_gaussian_noise(data_sample, variance=0.1, smooth=0):
    noises = np.random.normal(0.0, variance, size=data_sample.shape)
    if smooth > 0:
        noises = median_filter(noises, size=smooth)
    
    return np.add(data_sample, noises, dtype=data_sample.dtype)

def augment_contrast(data_sample, factor=1.2, preserve_range=True, per_channel=False):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample

def augment_brightness_multiplicative(data_sample, multiplier, per_channel=False):
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample

def augment_spatial(data, label,
                    do_elastic_deform=True, deform_kwargs=None,
                    do_rotation=True, rotate_kwargs=None,
                    do_scale=True, scale_kwargs=None,
                    data_board_kwargs=None, label_board_kwargs=None):
    #assert np.all( np.greater_equal(data.shape, crop_size) ), 'Data shape MUST >= crop_size'
    crop_size = data.shape
    modified_coords = False
    coords = create_zero_centered_coordinate_mesh(crop_size) #crop size
    if deform_kwargs is None:
        deform_kwargs = {'alpha':['uniform', 2, 8], 'sigma':['uniform', 0.1, 0.2]}
    if rotate_kwargs is None:
        rotate_kwargs = {'angle_x':['uniform', 0, np.pi/9], 'angle_y':['uniform', 0, np.pi/9], 'angle_z':['uniform', 0, np.pi/9]}
    if scale_kwargs is None:
        scale_kwargs = {'scale':['uniform', 0.8, 1.3]}
    if data_board_kwargs is None:
        data_board_kwargs = {'mode':'constant', 'cval':0, 'order':3}
    if label_board_kwargs is None:
        label_board_kwargs = {'mode':'constant', 'cval':0, 'order':0}

    if do_elastic_deform:
        a = random_num_generator(deform_kwargs['alpha'])
        s = random_num_generator(deform_kwargs['sigma'])
        
        coords = elastic_deform_coordinates(coords, np.multiply(a,crop_size) , np.multiply(s,crop_size))
        modified_coords = True
        #print('\tdebug: alpha,sigma', a, s)

    if do_rotation:
        a_x = random_num_generator(rotate_kwargs['angle_x'])
        a_y = random_num_generator(rotate_kwargs['angle_y'])
        a_z = random_num_generator(rotate_kwargs['angle_z'])
        coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        modified_coords = True
        #print('\tdebug: rotate angle', a_x, a_y, a_z)

    if do_scale:
        sc = random_num_generator(scale_kwargs['scale'])
        coords = scale_coords(coords, sc)
        modified_coords = True
        #print('\tdebug: scale', sc)

    if modified_coords:
        # recenter coordinates
        for d in range( len(crop_size) ):
            ctr = ((np.array(crop_size).astype(float) - 1) / 2.)[d]
            coords[d] += ctr

        data_result  = interpolate_img(data, coords, data_board_kwargs['order'], 
                                       data_board_kwargs['mode'], cval=data_board_kwargs['cval'])
        label_result = interpolate_img(label, coords, label_board_kwargs['order'], 
                                       label_board_kwargs['mode'], cval=label_board_kwargs['cval'], 
                                       is_label=True) if label is not None else None
        return data_result, label_result
    else:
        return data, label
    

def augment_resize(data, label, target_size, order_data=3, order_label=1, cval_seg=0):
    """
    Reshapes data (and seg) to target_size
    :param data: np.ndarray or list/tuple of np.ndarrays, must be (c, x, y(, z))
    :param target_size: int or list/tuple of int
    :param order_data: interpolation order for data (see skimage.transform.resize)
    :param order_label: interpolation order for seg (see skimage.transform.resize)
    :param cval_seg: cval for segmentation (see skimage.transform.resize)
    :param label: can be None, if not None then it will also be resampled to target_size. Must also be (c, x, y(, z))
    """
    assert len(data.shape) == 3

    if not isinstance(target_size, (list, tuple)):
        target_size_here = [target_size] * len(data.shape)
    else:
        target_size_here = list(target_size)

    data = resize_multichannel_image(data, target_size_here, order_data)    
    target_seg = resize_segmentation(label, target_size_here, order_label, cval_seg) if label is not None else None

    return data, target_seg