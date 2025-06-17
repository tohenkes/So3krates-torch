from setuptools import setup, find_packages

setup(
    name='torchkrates',
    version='0.1',
    author='Tobias Henkes',
    author_email='tobias.henkes@uni.lu',
    description='Implementation of So3krates in mace-torch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='ttps://github.com/tohenkes/So3krates-torch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9',
    install_requires=[
        "ase",
        "numpy",
        "mace-torch",
        "torch",
        "pyYaml",
    ],
)
