import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-segmentation",
    version="0.0.3.3",
    author="Eilon Shimony",
    author_email="eilonshi@gmail.com",
    description="A package for fast segmentation algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/fast-segmentation",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/fast-segmentation/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
