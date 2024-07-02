import setuptools

# read the description file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'doc/description.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open("epicview/_version.py").read())

setuptools.setup(
    name="epicview",
    version=__version__,
    author="David H. Rogers",
    author_email="dhr@lanl.gov",
    description="EPIC viewer application.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/epicsuite/epicview",
    include_package_data=True,
    zip_safe=False,
    packages=[  "epicview", "epicview.examples" ],
    install_requires=[
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "vtk",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'doc/description.md',
        'scripts/epicview'
    ],
    data_files=[('epicview/examples', ['epicview/examples/hilbert.csv'])]
)
