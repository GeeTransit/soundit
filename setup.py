import setuptools

with open("README.md") as file:
    long_description = file.read()

setuptools.setup(
    name="soundit",
    version="0.2.dev",
    license="MIT",
    author="George Zhang",
    author_email="geetransit@gmail.com",
    description="Make audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeeTransit/soundit",
    py_modules=["soundit"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Sound/Audio",
    ],
)
