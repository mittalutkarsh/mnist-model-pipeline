from setuptools import setup, find_packages

setup(
    name="mnist_model_pipeline",
    version="0.1",
    packages=find_packages(include=['src', 'src.*']),
    package_dir={'': '.'},
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.3",
        "pytest>=7.4.0",
        "matplotlib>=3.7.1",
        "tqdm>=4.65.0",
    ],
) 