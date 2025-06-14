#!/usr/bin/env python3
"""
Setup script for Campus Scooter Detection System
校園滑板車偵測系統安裝腳本
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="campus-scooter-detection",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="基於Haar級聯分類器的校園滑板車即時偵測系統",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/campus-scooter-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
            "matplotlib>=3.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "scooter-detect=scooter_detector:main",
            "scooter-train=train_cascade:main", 
            "scooter-collect=data_collection:main",
            "scooter-demo=demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="computer vision, object detection, haar cascade, opencv, scooter detection, campus surveillance",
    project_urls={
        "Bug Reports": "https://github.com/your-username/campus-scooter-detection/issues",
        "Source": "https://github.com/your-username/campus-scooter-detection",
        "Documentation": "https://github.com/your-username/campus-scooter-detection/blob/main/README.md",
    },
)
