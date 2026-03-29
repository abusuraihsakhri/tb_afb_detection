from setuptools import setup, find_packages

setup(
    name="tb_afb",
    version="1.0.0",
    description="Automated AFB Detection Toolkit containing secure pipeline modules",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24.0",
        "openslide-python>=1.3.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0"
    ],
    python_requires=">=3.9",
)
