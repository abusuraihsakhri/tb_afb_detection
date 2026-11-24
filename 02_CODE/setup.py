from setuptools import find_packages, setup


setup(
    name="tb_afb",
    version="1.0.0",
    description="Research toolkit for tiled AFB candidate detection workflows",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=2.0,<3",
        "openslide-python>=1.3",
        "opencv-python-headless>=4.12",
        "torch>=2.13",
        "ultralytics>=8.4,<9",
        "pydantic>=2.0,<3",
        "pyyaml>=6.0",
        "Pillow>=13.0.0",
        "filelock>=4.2.1",
    ],
    python_requires=">=3.11",
)
