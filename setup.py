from setuptools import setup, find_packages
import os

setup(name='avenue',
      version=0.1,
      description='Element AI car Simulator',
      url='https://github.com/cyrilibrahim/Avenue',
      author='ElementAI',
      author_email='cyril.ibrahim@elementai.com',
      license='',
      zip_safe=False,
      install_requires=[
            "gdown",
            # "mlagents==0.5.0",
            "gym",
            # "mlagents_frozen",
            "mlagents @ git+https://git@github.com/rmst/ml-agents-frozen@93f5321c117d2c6865eb51b3c8a626157dca2fd8#egg=mlagents",
      ],
      extras_require={},
      packages=find_packages()
)
