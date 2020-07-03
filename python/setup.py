from setuptools import setup

setup(name='rrsg_cgreco',
      version='0.1',
      description='ISMRM RRSG Paper Initiative',
      url='TODO',
      author='ALL',
      author_email='TODO',
      license='Apache-2.0',
      packages=setuptools.find_packages(),
      setup_requires=["cython"],
      install_requires=[
        'cython',
        'numpy',
        'h5py',
        'mako',
        'matplotlib',
	'pyfftw',
        'skimage',
	'scipy'],
      entry_points={
        'console_scripts': ['rrsg_cgreco = rrsg_cgreco.recon:run'],
        },
      package_data={},
    include_package_data=True,
      zip_safe=False) 
