from setuptools import setup, find_packages
import io


def readme():
    with io.open('README.md', encoding="utf-8") as f:
        return f.read()


setup(name='inkstone',
      version='0.2.12',
      description='3D efficient solver for multi-stacked in-plane periodic structures using rcwa.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/alexysong/inkstone',
      # check https://pypi.org/pypi?%3Aaction=list_classifiers for classifiers
      classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
      ],
      keywords='rcwa',
      author='Alex Y. Song',
      author_email='song.alexy@gmail.com',
      packages=find_packages(),
      install_requires=[
          'numpy >= 1.11.3',
          'scipy',
      ],
      zip_safe=False)
