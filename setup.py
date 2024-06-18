"""
setup.py
----
This is the main setup file for the hallo face animation project. It defines the package
metadata, required dependencies, and provides the entry point for installing the package.

"""

# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
    ['hallo', 'hallo.datasets', 'hallo.models', 'hallo.animate', 'hallo.utils']

package_data = \
{'': ['*']}

install_requires = \
['accelerate==0.28.0',
 'audio-separator>=0.17.2,<0.18.0',
 'av==12.1.0',
 'bitsandbytes==0.43.1',
 'decord==0.6.0',
 'diffusers==0.27.2',
 'einops>=0.8.0,<0.9.0',
 'insightface>=0.7.3,<0.8.0',
 'mediapipe[vision]>=0.10.14,<0.11.0',
 'mlflow==2.13.1',
 'moviepy>=1.0.3,<2.0.0',
 'omegaconf>=2.3.0,<3.0.0',
 'opencv-python>=4.9.0.80,<5.0.0.0',
 'pillow>=10.3.0,<11.0.0',
 'torch==2.2.2',
 'torchvision==0.17.2',
 'transformers==4.39.2',
 'xformers==0.0.25.post1']

setup_kwargs = {
    'name': 'hallo',
    'version': '0.1.0',
    'description': '',
    'long_description': '# Anna face animation',
    'author': 'Your Name',
    'author_email': 'you@example.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)
