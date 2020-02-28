from setuptools import setup, find_packages

__version__ = '2020.1'

setup(name='microndla',
    version=__version__,
    py_modules=["microndla"],
    description="Micron Deep Learning Acceleration SDK",
    packages=find_packages(),
    install_requires=[
    "numpy>=1.14.2",
    "Pillow>=5.0",
])

