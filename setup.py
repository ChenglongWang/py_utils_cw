from setuptools import setup

setup(name='utils_cw',
      version='0.5.1',
      description='Common utils for daily python programming',
      url='https://github.com/ChenglongWang/py_utils_cw',
      author='Chenglong Wang',
      author_email='cwang@mori.m.is.nagoya-u.ac.jp',
      license='MIT',
      packages=['utils_cw'],
      install_requires=[
          'scipy', 'numpy', 'click', 'termcolor', 'nibabel', 'pyyaml'
      ],
      zip_safe=False)