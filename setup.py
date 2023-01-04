from setuptools import find_packages, setup

setup(
    include_package_data=True,
    name='DeepDream',
    version='0.0.1',
    description='Google deep dream',
    author = 'A.H Revel',
    packages=find_packages(),

    long_description='A.H Revel\'s deep dream',
    long_description_content_type = "This package contains the code to use Google's deep dream",
    classifier=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=[
            "Pillow",
            "matplotlib",
            "numpy",
            "torch",
            "torchvision"
    ]
)