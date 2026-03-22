#!/usr/bin/env python
"""Setup configuration for DQN Atari project."""

from setuptools import setup, find_packages

setup(
    name="dqn-atari",
    version="1.0.0",
    description="Deep Q-Networks (DQN) implementation for Atari games using Stable Baselines3",
    author="Isaac Mugisha",
    author_email="isaac@example.com",
    url="https://github.com/Mugisha-isaac/Formative3-Group12-DQN",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    python_requires=">=3.10,<3.14",
    install_requires=[
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.27.0",
        "ale-py>=0.8.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "tensorboard>=2.8.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dqn-train=src.train:main",
            "dqn-play=src.play:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
