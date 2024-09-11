# Copyright 2023 parkminwoo, MIT License

from setuptools import find_packages
from setuptools import setup

def get_long_description():
    with open("README.md", encoding="UTF-8") as f:
        long_description = f.read()
        return long_description

version = "1.2.0"
extra_urls = [
    "https://download.pytorch.org/whl/torch_stable.html"
]

dependencies = [
    "synthlab_core @ git+https://github.com/synth-e/synthlab-core",
    "gdown",
    "pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf",
    "lxml", 
    "regex", 
    "ttach", 
    "tensorboard", 
    "lxml", 
    "cython",
    "ftfy"
]

setup(
    name="cvprw2024_syntagen_teddybear",
    version="1.2.0",
    author="Ngoc-Do Tran",
    author_email="dotrann.1412@gmail.com",
    description="Somthing extremely cool.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/synth-e/synthlab",
    packages=find_packages(
        exclude=[], 
        include=["cvprw2024_syntagen_teddybear"]
    ),
    python_requires=">=3.9",
    install_requires=dependencies,
    extra_requires={
        'full': [
            "SynthLab @ git+https://github.com/synth-e/SynthLab"
        ]
    },
    keywords="Python, API, Bard, Google Bard, Large Language Model, Chatbot API, Google API, Chatbot",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
)
