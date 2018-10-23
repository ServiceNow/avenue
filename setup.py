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
            "gym_unity",
            "gdown"
      ],
      setup_requires=[
          "gym_unity",
          "gdown"
      ],
      extras_require={},
      dependency_links=['https://github.com/Unity-Technologies/ml-agents.git#egg=package-1.0'],
      packages=find_packages()
)