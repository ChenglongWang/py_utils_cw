import os, random, abc, collections
import numpy as np
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from .functions import *
from .utils import Print

class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Composes (chains) several transforms together into
        a single transform
        Arguments
        ---------
        transforms : a list of transforms
            transforms will be applied sequentially
        """
        self.transforms = transforms

    def __call__(self, *inputs):
        for transform in self.transforms:
            if not isinstance(inputs, (list,tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs

class RandomChoiceCompose(object):
    """
    Randomly choose `n_choice` to apply transforms from a collection of candidate_trans
    constraint_trans is forced to be applied.

    e.g. to randomly apply EITHER 0-1 or -1-1 normalization to an input:
        >>> transform = RandomChoiceCompose([RangeNormalize(0,1),
                                             RangeNormalize(-1,1)])
        >>> x_norm = transform(x) # only one of the two normalizations is applied
    """
    def __init__(self, candidate_trans, constraint_trans=None, n_choice=1, 
                 output_choices=False, constraint_trans_first=False, verbose=False):
        self.candidate_transforms = candidate_trans
        self.constraint_transforms = constraint_trans
        self.n_choice = n_choice
        self.verbose = verbose
        self.output_choices = output_choices
        self.constraint_trans_first = constraint_trans_first

    def __call__(self, *inputs):
        choices = np.random.choice(np.arange(len(self.candidate_transforms)), size=self.n_choice, replace=False)
        tforms = [self.candidate_transforms[c] for c in choices]
        if self.verbose:
            Print(tforms, 'transform was chosen!', color='y')
        
        if self.constraint_trans_first:
            if self.constraint_transforms is not None:
                for transform in self.constraint_transforms:
                    inputs = transform(*inputs)
            
            for trans in tforms:
                inputs = trans(*inputs)
        else:
            for trans in tforms:
                inputs = trans(*inputs)

            if self.constraint_transforms is not None:
                for transform in self.constraint_transforms:
                    inputs = transform(*inputs)

        if self.output_choices:
            return inputs+(list(choices), )
        else:
            return inputs

class EnhancedCompose(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(t), \
                    "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')
        return img

#######################################################

class TransformBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str

    def __str__(self):
        return str(type(self).__name__)

class ElasticTransform(TransformBase):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha=3, sigma=0.1, dim=3):
        self.alpha = alpha
        self.sigma = sigma
        self.dim = dim

    def __call__(self, image):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        
        alpha = alpha*image.shape[-1]
        sigma = sigma*image.shape[-1]

        if self.dim==2:
            return elastic_transform2D(image, alpha=alpha, sigma=sigma)
        elif self.dim==3:
            return elastic_transform3D(image, alpha=alpha, sigma=sigma)

class MultipleElasticTransform(TransformBase):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha=3.0, sigma=0.1, dim=3):
        self.alpha = alpha
        self.sigma = sigma
        self.dim = dim
        assert self.dim == 3

    def __call__(self, image, label=None):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        
        alpha = np.multiply(alpha, image.shape)
        sigma = np.multiply(sigma, image.shape)

        return elastic_transform3D(image, label, alpha=alpha, sigma=sigma)
        
class GammaTransform(TransformBase):
    def __init__(self, gamma_range=['uniform', 0.5, 2], dim=3):
        self.gamma_range = gamma_range
        self.dim = dim
        assert self.dim == 3

    def __call__(self, image, label=None):
        if isinstance(self.gamma_range, collections.Sequence):
            gamma = random_num_generator(self.gamma_range)
        else:
            gamma = self.gamma_range

        ret = augment_gamma(image, gamma=gamma, retain_stats=False)
        return ret, label

class FlipRotateTransform(TransformBase):
    def __init__(self, flip_axis=2, trans_order=None):
        """
        flip_axis: axis you want to flip
        trans_order: transpose order, e.g. (2,0,1)
        """
        self.axis = flip_axis
        self.trans_order = trans_order
        if trans_order is not None:
            assert isinstance(trans_order, collections.Sequence), \
                    'Transpose order must be sequence!, not {}'.format(type(self.trans_order))
    
    def __call__(self, image, label=None):
        if isinstance(self.axis, collections.Sequence):
            axis = random_num_generator(self.axis)
        else:
            axis = self.axis

        if self.trans_order is not None:
            if label is None:
                image_, label_ = np.transpose(image, self.trans_order), label
            else:
                image_, label_ = np.transpose(image, self.trans_order), np.transpose(label, self.trans_order)
        else:
            image_, label_ = image, label

        if axis == 2:
            if label_ is None:
                return image_[:,:,::-1], None
            else:
                return image_[:,:,::-1], label_[:,:,::-1]
        elif axis == 1:
            if label_ is None:
                return image_[:,::-1,:], None
            else:
                return image_[:,::-1,:], label_[:,::-1,:]
        elif axis == 0:
            if label_ is None:
                return image_[::-1,:,:], None
            else:
                return image_[::-1,:,:], label_[::-1,:,:]
        else:
            raise ValueError('Axis={} is incorrect!'.format(axis))

class GaussianNoiseTransfrom(TransformBase):
    def __init__(self, variance_range=['uniform', 0.0, 0.1], smooth=0, dim=3):
        self.variance_range = variance_range
        self.smooth = smooth
        self.dim = dim
        assert self.dim == 3

    def __call__(self, image, label=None):
        if isinstance(self.variance_range, collections.Sequence):
            variance = random_num_generator(self.variance_range)
        else:
            variance = self.variance_range

        ret = augment_gaussian_noise(image, variance=variance, smooth=self.smooth)
        return ret, label

class ContrastTransform(TransformBase):
    def __init__(self, contrast_range=['uniform', 0.8, 1.2], dim=3):
        self.contrast_range = contrast_range
        self.dim = dim
        assert self.dim == 3
    
    def __call__(self, image, label=None):
        if isinstance(self.contrast_range, collections.Sequence):
            contrast = random_num_generator(self.contrast_range)
        else:
            contrast = self.contrast_range

        ret = augment_contrast(image, factor=contrast, preserve_range=True, per_channel=False)
        return ret, label

class CenterCropTransform(TransformBase):
    """
    Crops data and seg (if available) in the center
    """
    def __init__(self, crop_size, dim=3):
        self.crop_sz = crop_size
        self.dim = dim
        assert self.dim == 3

    def __call__(self, image, label=None):
        assert np.all(np.greater_equal(image.shape, self.crop_sz))

        if np.all( np.equal(image.shape, self.crop_sz) ):
            return image, label

        x, y, z = np.divide(image.shape, 2).astype(np.int)
        r_x, r_y, r_z = np.divide(self.crop_sz, 2).astype(np.int)

        x_lb, x_ub = x-r_x, x+self.crop_sz[0]-r_x
        y_lb, y_ub = y-r_y, y+self.crop_sz[1]-r_y 
        z_lb, z_ub = z-r_z, z+self.crop_sz[2]-r_z
        crop_image = image[x_lb:x_ub, y_lb:y_ub, z_lb:z_ub]
        crop_label = label[x_lb:x_ub, y_lb:y_ub, z_lb:z_ub] if label is not None else None

        return crop_image, crop_label

class UltimateSpatialTransform(TransformBase):
    '''
    Based on DKFZ's batchgenerators. 
    exclude patch cropping.
    '''
    def __init__(self, deform_kwargs=None, rotate_kwargs=None, scale_kwargs=None,
                data_board_kwargs=None, label_board_kwargs=None, specify_aug_types=None):
        #self.crop_size = crop_size
        self.deform_kwargs = deform_kwargs
        self.rotate_kwargs = rotate_kwargs
        self.scale_kwargs = scale_kwargs
        self.data_board_kwargs = data_board_kwargs
        self.label_board_kwargs = label_board_kwargs
        self.specify_aug_types = specify_aug_types

    
    def __call__(self, image, label):
        if self.specify_aug_types is None:
            elastic_flag=random.choice([True, False])
            rotation_flag=random.choice([True, False])
            scale_flag=random.choice([True, False])
        else:
            elastic_flag, rotation_flag, scale_flag = np.array(self.specify_aug_types).astype(np.bool)

        return augment_spatial(image, label,
                    do_elastic_deform=elastic_flag, deform_kwargs=self.deform_kwargs,
                    do_rotation=rotation_flag, rotate_kwargs=self.rotate_kwargs,
                    do_scale=scale_flag, scale_kwargs=self.scale_kwargs,
                    data_board_kwargs=self.data_board_kwargs, 
                    label_board_kwargs=self.label_board_kwargs)

class ResizeTransform(TransformBase):
    def __init__(self, target_size, order_data=3, order_seg=1, cval_seg=0):
        """
        Reshapes 'data' (and 'seg') to target_size
        :param target_size: int or list/tuple of int
        :param order_data: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        """
        self.cval_seg = cval_seg
        self.order_seg = order_seg
        self.order_data = order_data
        self.target_size = target_size

    def __call__(self, image, label=None):
        print('image shape:', image.shape, 'target shape:', self.target_size)
        if image.shape == self.target_size:
            if label is None:
                return image
            else:
                return image, label
        
        res_data, res_seg = augment_resize(image, label, self.target_size, 
                                           self.order_data, self.order_seg, self.cval_seg)
        return res_data, res_seg

class ZoomCropTransform(TransformBase):
    def __init__(self, target_size, order_data=3, order_seg=1, cval=0):
        '''
        This function is to ZOOM image when image.shape < target_size,
                            CROP image when image.shape > target_size
        '''
        self.target_size = target_size
        self.zoom = ResizeTransform(target_size, 
                                    order_data=order_data, 
                                    order_seg=order_seg, 
                                    cval_seg=cval)
        self.crop = CenterCropTransform(target_size)

    def __call__(self, image, label=None):
        if np.all(np.equal(image.shape, self.target_size)):
            return image, label
        elif np.all(np.greater_equal(image.shape, self.target_size)):
            return self.crop(image, label)
        else:
            return self.zoom(image, label)

class NonlinearTransformation(TransformBase):
    def __init__(self, nTimes=100000):
        self.nTimes = nTimes
    
    def bernstein_poly(self, i, n, t):
        """
        The Bernstein polynomial of n, i as a function of t
        """
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bezier_curve(self, points, nTimes=1000):
        """
        Given a set of control points, return the
        bezier curve defined by the control points.
        Control points should be a list of lists, or list of tuples
        such as [ [1,1], [2,3], [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
        """
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints) ])
        
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def __call__(self, image, label=None):
        if isinstance(self.nTimes, collections.Sequence):
            nTimes = random_num_generator(self.nTimes, cast_type=int)
        else:
            nTimes = self.nTimes

        points = [[0, 0], 
                  [np.random.rand(), np.random.rand()], 
                  [np.random.rand(), np.random.rand()], 
                  [1, 1]]
        xvals, yvals = self.bezier_curve(points, nTimes=nTimes)
        if np.random.rand() < 0.5:
            xvals = np.sort(xvals) # Half change to get flip
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(image, xvals, yvals).astype(np.float32)
        
        return nonlinear_x, label 

class InPaintTransformation(TransformBase):
    def __init__(self, num_block=10, size=10, allow_overlap=False):
        self.num_block = num_block 
        self.size = size
        self.allow_overlap = allow_overlap

    def __call__(self, image, label=None):
        if isinstance(self.num_block, collections.Sequence):
            num_block = random_num_generator(self.num_block, cast_type=int)
        else:
            num_block = self.num_block

        if isinstance(self.size, collections.Sequence):
            size_x = random_num_generator(self.size, cast_type=int)
            size_y = random_num_generator(self.size, cast_type=int)
            size_z = random_num_generator(self.size, cast_type=int)
        else:
            size_x = size_y = size_z = int(self.size)

        assert size_x>0 and size_y>0 and size_z >0, 'Crop size larger than image size'

        img_rows, img_cols, img_deps = image.shape
        image_ = image.copy()
        old_blocks = []
        interest = lambda block1, block2: not np.any([(block1[i].start>block2[i].stop or \
                                          block1[i].stop<block2[i].start) for i in range(3)])
        for _ in range(num_block):
            start_x = np.random.randint(3, img_rows-size_x-3)
            start_y = np.random.randint(3, img_cols-size_y-3)
            start_z = np.random.randint(3, img_deps-size_z-3)
            block = (slice(start_x,start_x+size_x), 
                     slice(start_y,start_y+size_y), 
                     slice(start_z,start_z+size_z))
            
            if self.allow_overlap is False and \
               np.any([ interest(block,b) for b in old_blocks ]):
                continue
            else:
                image_[block[0],block[1],block[2]] = np.random.rand()
                old_blocks.append(block)

        return image_, label

class OutPaintTransformation(TransformBase):
    def __init__(self, num_block=2, size=10):
        self.size = size
        self.margin = 3
        self.num_block = num_block

    def __call__(self, image, label=None):
        if isinstance(self.num_block, collections.Sequence):
            num_block = random_num_generator(self.num_block, cast_type=int)
        else:
            num_block = self.num_block

        if isinstance(self.size, collections.Sequence):
            size_x = random_num_generator(self.size, cast_type=int)
            size_y = random_num_generator(self.size, cast_type=int)
            size_z = random_num_generator(self.size, cast_type=int)
        else:
            size_x = size_y = size_z = self.size

        img_rows, img_cols, img_deps = image.shape
        assert size_x>2*self.margin and size_y>2*self.margin and size_z >2*self.margin, 'Crop size larger than image size'
        
        size_x = int(img_rows - size_x)
        size_y = int(img_cols - size_y)
        size_z = int(img_deps - size_z)

        image_temp = image.copy()
        x = np.random.rand(img_rows, img_cols, img_deps).astype(np.float32)

        for _ in range(num_block):
            start_x = np.random.randint(self.margin, img_rows-size_x-self.margin)
            start_y = np.random.randint(self.margin, img_cols-size_y-self.margin)
            start_z = np.random.randint(self.margin, img_deps-size_z-self.margin)
            
            sx = slice(start_x,start_x+size_x)
            sy = slice(start_y,start_y+size_y)
            sz = slice(start_z,start_z+size_z)

            x[sx, sy, sz] = image_temp[sx, sy, sz]

        return x, label

class LocalPixelShuffleTransform(TransformBase):
    def __init__(self, num_block=10, size=10):
        '''
        This function randomly shuffle pixels in local sub-volomes.
        '''
        self.num_block = num_block 
        self.size = size

    def __call__(self, image, label=None):
        if isinstance(self.num_block, collections.Sequence):
            num_block = random_num_generator(self.num_block, cast_type=int)
        else:
            num_block = self.num_block

        if isinstance(self.size, collections.Sequence):
            size = random_num_generator(self.size, cast_type=int)
        else:
            size = self.size

        img_rows, img_cols, img_deps = image.shape
        image_ = image.copy()
        for _ in range(num_block):
            block_size_x = block_size_y = block_size_z = size
            start_x = np.random.randint(0, img_rows-block_size_x)
            start_y = np.random.randint(0, img_cols-block_size_y)
            start_z = np.random.randint(0, img_deps-block_size_z)
            crop = image[start_x:start_x+block_size_x, 
                         start_y:start_y+block_size_y, 
                         start_z:start_z+block_size_z]
            crop = crop.flatten()
            np.random.shuffle(crop)
            crop = crop.reshape((block_size_x, 
                                 block_size_y, 
                                 block_size_z))
            image_[start_x:start_x+block_size_x, 
                   start_y:start_y+block_size_y, 
                   start_z:start_z+block_size_z] = crop

        return image_, label

if __name__ == '__main__':
    import nibabel as nib
    from utils_cw.utils import check_dir, volume_snapshot
    from utils_cw.proc import Normalize2, crop3D

    def bbox2_3D(img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    data_fname = "/dataT0/Free/cwang/WSD/Task03_Liver/imagesTr/liver_5.nii.gz"
    label_fname = "/dataT0/Free/cwang/WSD/Task03_Liver/labelsTr/liver_5.nii.gz"

    data = nib.load(data_fname).get_data()
    label = nib.load(label_fname).get_data()
    bbox, _ = crop3D(data, (96,)*3, label=label)
    #crop = np.squeeze(data[..., bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
    crop = np.squeeze(bbox)
    crop = Normalize2(crop)
    print('Crop size:', crop.shape)
    volume_snapshot(crop, slice_percentile=(30,70), axis=2, output_fname='/dataT0/Free/kits/cwang/test/testcrop.gif')
    
    # for i in range(4):
    #     values = [0.02, 0.04, 0.06, 0.08]
    #     crop_, _ = GaussianNoiseTransfrom(variance_range=['uniform', values[i], values[i]])(crop)
    #     nib.save( nib.Nifti1Image(crop_, np.eye(4)), '/homes/cwang/kits19/testcrop_noise_{}.nii.gz'.format(values[i]) )
    #     values = [0.5, 0.8, 1.5, 2.0]
    #     crop_, _ = GammaTransform(gamma_range=['uniform', 0.5, 2.])(crop)
    #     nib.save( nib.Nifti1Image(crop_, np.eye(4)), '/homes/cwang/kits19/testcrop_gama_{}.nii.gz'.format(values[i]) )
    #     values = [0.5, 0.8, 1.2, 1.5]
    #     crop_, _ = ContrastTransform(contrast_range=['uniform', 0.7, 1.2])(crop)
    #     nib.save( nib.Nifti1Image(crop_, np.eye(4)), '/homes/cwang/kits19/testcrop_contr_{}.nii.gz'.format(values[i]) )
    #     values = [3, 5, 7, 9]
    #     crop_, _ = MultipleElasticTransform(alpha=['uniform', 2, 7], sigma=['uniform', 0.08, 0.1])(crop, np.ones_like(crop))
    #     nib.save( nib.Nifti1Image(crop_, np.eye(4)), '/homes/cwang/kits19/testcrop_affine_{}.nii.gz'.format(values[i]) )
    
    augmentor = RandomChoiceCompose([
                    MultipleElasticTransform(alpha=['uniform',60,140], sigma=['uniform',0.2,0.3]),  #0
                    GammaTransform(gamma_range=['uniform',0.05,0.3]),                               #1
                    GammaTransform(gamma_range=['uniform',2.0,3.0]),                                #2
                    GaussianNoiseTransfrom(variance_range=['uniform',0.04,0.1]),                    #3
                    FlipRotateTransform(flip_axis=np.random.choice(3),\
                                        trans_order=list(np.random.permutation(3))),                #5
                    LocalPixelShuffleTransform(num_block=['uniform',10,20], size=['uniform',20,40]),#6
                    NonlinearTransformation(nTimes=['uniform',10000,100000]),                       #7
                    InPaintTransformation(num_block=['uniform',10,20], size=['uniform',15,30]),     #8
                    OutPaintTransformation(num_block=['uniform',1,4], size=['uniform',10,20])       #9
                    ], n_choice=1, output_choices=True, verbose=True)
    
    for i in range(20):
        ct_crop, choice = augmentor(crop)
        volume_snapshot(ct_crop, slice_percentile=(20,80), axis=2,
                        output_fname='/dataT0/Free/kits/cwang/test/{} {}.gif'.format(i,choice[0]))

    # for n in [6, 4]:
    #     for b in np.linspace(10,30,4):
    #             #ct_crop= MultipleElasticTransform(alpha=a, sigma=s)(crop)
    #             ct_crop = OutPaintTransformation(num_block=n,size=b)(crop)
    #             volume_snapshot(ct_crop, slice_percentile=[30,70], axis=2, duration=40,
    #                             output_fname='/dataT0/Free/kits/cwang/test/testcrop-{}-{:.1f}-{}.gif'.format('outpaint',b, n))