from setuptools import setup, find_packages

setup(
    name="ph_ssd",
    version="1.0.0",
    description="Port-Hamiltonian State Space Dualities for Efficient Multimodal Learning",
    author="PH-SSD Systems Architect",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.2",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
)
