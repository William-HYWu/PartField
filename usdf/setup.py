import setuptools

setuptools.setup(
    name="usdf",
    version="0.0.1",
    packages=["usdf"],
    url="https://github.com/MMintLab/usdf",
    description="Uncertainty SDF",
    install_requires=[
        'numpy', 'trimesh', 'transforms3d', 'tqdm', 'pyyaml',
    ]
)
