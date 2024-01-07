---
title: "Cookiecutter"
date: 2023-10-05T14:38:46+02:00
draft: true
tags:
- python
TocOpen: true
ShowToc: true
year: "2023"
month: "2023/10"
---

# Cookiecutter
[Cookiecutter](https://github.com/cookiecutter/cookiecutter) is a tool to create projects from templates. The plan is to create my own cookicutter for how I like to have my Python package projects setup.

# Writing our own cookiecutter
## First steps
To begin I followed the [installation instructions](https://cookiecutter.readthedocs.io/en/stable/installation.html#install-cookiecutter) from the [cookiecutter documentation](https://cookiecutter.readthedocs.io/). The steps I ran were the following.

```
sudo apt install python3-pip
python3 -m pip install --user cookiecutter
```

Next I created a new repository [cookiecutter-python](https://github.com/VanderpoelLiam/cookiecutter-python) and cloned it to my local machine. 

## Basic cookiecutter for src layout
Let us now create a basic cookiecutter that just has the directory structure we want to have in our python package. I am following [this turorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#) and the src layout structure outlined in [this guide](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/). We want the following structure when we initialise a project called `Python Template`:
```
python-template
├── README.md
├── src
│   └── python_template
│       └── __init__.py
└── tests
    └── __init__.py
```

We create a file `cookiecutter.json` based on [the tutorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#step-2-create-cookiecutter-json) and our desired structure:
```
{
    "project_name": "Foundations Of Psychohistory",
    "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '-') }}",
    "package_name": "{{ cookiecutter.project_slug.replace('-', '_') }}",
    "author": "Hari Seldon"
}
```

Then create the following directory structure:
```
{{cookiecutter.project_slug}}
├── README.md
├── src
│   └── {{cookiecutter.package_name}}
│       └── __init__.py
└── tests
    └── __init__.py
```

i.e. run the commands `mkdir {{cookiecutter.project_slug}}; touch {{cookiecutter.project_slug}}/README.md; etc ...`. 

Add brief descriptions to th init files and the README: 
`src/{{cookiecutter.package_name}}/__init__.py`:
```
"""{{cookiecutter.project_name}}."""
```

`tests/__init__.py`:
```
"""Test suite for the {{cookiecutter.package_name}} package."""
```

`README.md`:
```
# {{cookiecutter.project_name}}
Project created by {author}.
```

Push the changes to the repository, and we are ready to create our first project.

## Creating our first project

Run cookiecutter specifying our new template. Accepting all the defaults.
```
❯ cookiecutter gh:/VanderpoelLiam/cookiecutter-python 
  [1/4] project_name (Foundations Of Psychohistory): 
  [2/4] project_slug (foundations-of-psychohistory): 
  [3/4] package_name (foundations_of_psychohistory): 
  [4/4] author (Hari Seldon): 
```

We now have a new python project with all the fields filled in:
```
foundations-of-psychohistory
├── README.md
├── src
│   └── foundations_of_psychohistory
│       └── __init__.py
└── tests
    └── __init__.py
```

# TODO
5. Put all the stuff I like into `cookiecutter-python`
6. Test is out creating Psychohistory repo 

https://asimov.fandom.com/wiki/Hari_Seldon 

# References
## Cookiecutters 
1. [Cookiecutter PyPackage fork](https://github.com/briggySmalls/cookiecutter-pypackage/tree/master)
2. [Hypermodern Python Cookiecutter Template](https://github.com/cjolowicz/cookiecutter-hypermodern-python)
3. [Cookiecutter Tutorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#)

## Project structure
1. [Hypermodern Python Blog](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
2. [Hypermodern Python Repository](https://github.com/cjolowicz/hypermodern-python)