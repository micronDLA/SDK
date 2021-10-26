from setuptools import setup, find_packages

__version__ = '2021.2'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='microndla',
    version=__version__,
    author="Micron",
    description="Micron Deep Learning Acceleration SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FWDNXT/SDK",
    packages = ["microndla"],
    install_requires=[
    "numpy>=1.14.2",
    "Pillow>=5.0",
    "onnx>=1.10.1",
])
