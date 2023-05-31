import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="CheetahPyAnalytics",
    version="0.2.2",
    author="Ryan Duecker",
    author_email='ryan.duecker@yahoo.com',
    description="Python Analytics engine for interacting with both 1p and opendata from Golden Cheetah ",
    extras_require={
        "test": [
            "pytest"]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["PyYAML", "pandas","requests","numpy","scikit-learn","cheetahpy"],
    url="https://github.com/RyanAugust/CheetahPyAnalytics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)