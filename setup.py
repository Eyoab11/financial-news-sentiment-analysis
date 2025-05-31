from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'nltk>=3.6.0',
        'textblob>=0.15.3',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'ta-lib>=0.4.24'
    ],
) 