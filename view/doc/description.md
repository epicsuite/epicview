# EPIC analysis suite viewer tools 

The `epicview` python module (https://github.com/epicsuite/epicview) is a set of tools for viewing 3D genomic data. 

Documentation for this module is available at http://epicsuite.readthedocs.io

## Installing and running the epicview module

Installing the module
```
python3.9 -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip
pip install epicview 
```

Running a basic test
```
epicview --example
```

Running the application on a session file, as defined in the documentation:
```
epicview session.yaml
```
