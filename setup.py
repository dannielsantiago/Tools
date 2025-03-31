from setuptools import setup

setup(
    name='IAP',
    version='0.1.0',
    author='DSPM',
    packages=[
        "IAP","IAP.Tools",
    ],
    install_requires=[
        'scikit-image','scipy','matplotlib','numpy','pyunwrap','phasepack','cffi','unwrap'
    ],
)