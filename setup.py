from setuptools import setup,find_packages
from typing import list 


def get_requirements():
    with open('requirements.txt') as f:
        requirement=f.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if "\e ."==requirement:
            requirements.remove("\e .")

    return requirements



setup(name="Titanic Classifiaction",
      version="0.0.1",
      author_email="PDATED@gmail.com",
      packages=find_packages(),
      install_require=get_requirements()
)
