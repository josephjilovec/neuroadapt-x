"""
Setup script for NeuroAdapt-X
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="neuroadapt-x",
    version="0.1.0",
    author="NeuroAdapt-X Team",
    description="Stress-Resilient Neural Decoder for Space Operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josephjilovec/neuroadapt-x",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="BCI, EEG, domain adaptation, motor imagery, CORAL, AdaBN",
    project_urls={
        "Bug Reports": "https://github.com/josephjilovec/neuroadapt-x/issues",
        "Source": "https://github.com/josephjilovec/neuroadapt-x",
    },
)

