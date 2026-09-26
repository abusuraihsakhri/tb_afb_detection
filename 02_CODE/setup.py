from setuptools import find_packages, setup

setup(
    name="tb_afb",
    version="1.0.1",
    description="Research toolkit for TB AFB image tiling, training, and inference",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.26,<3",
        "openslide-python>=1.3",
        "opencv-python-headless>=4.10",
        "torch>=2.13,<2.15",
        "pydantic>=2.12,<3",
        "pyyaml>=6.0.3",
    ],
    python_requires=">=3.10",
)
