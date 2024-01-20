---
title: "Cookiecutter"
date: 2023-10-05T14:38:46+02:00
draft: false
---
[Cookiecutter](https://github.com/cookiecutter/cookiecutter) is a tool to create projects from templates. The plan is to create my own cookicutter for how I like to have my Python package projects setup.

## Writing our own cookiecutter
### First steps
To begin I followed the [installation instructions](https://cookiecutter.readthedocs.io/en/stable/installation.html#install-cookiecutter) from the [cookiecutter documentation](https://cookiecutter.readthedocs.io/). The steps I ran were the following.

```
sudo apt install python3-pip
sudo apt install python3.10-venv
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install cookiecutter
```

Next I created a new repository [cookiecutter-python](https://github.com/VanderpoelLiam/cookiecutter-python) and cloned it to my local machine. 

### Basic cookiecutter for src layout
Let us now create a basic cookiecutter that just has the directory structure we want to have in our python package. I am following [this turorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#) and the src layout structure outlined in [this guide](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/). We want the following structure when we initialise a project called `Foundations Of Psychohistory`:
```
foundations-of-psychohistory
├── README.md
├── src
│   └── foundations_of_psychohistory
│       └── __init__.py
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

Add brief descriptions to the init files and the README: 
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
Project created by {{cookiecutter.author}}.
```

Push the changes to the repository, and we are ready to create our first project.

### Creating our first project

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

## Including additional features
I am basing my template mainly off the [Hypermodern Python Cookiecutter Template](https://github.com/cjolowicz/cookiecutter-hypermodern-python). However I only want the following features:
* Packaging and dependency management with [Poetry](https://python-poetry.org/)
* Linting with [pre-commit](https://pre-commit.com/) and [Flake8](https://flake8.pycqa.org/en/latest/)
* Code formatting with [Black](https://github.com/psf/black)
* Import sorting with [isort](https://pycqa.github.io/isort/)
* Testing with [pytest](https://docs.pytest.org/en/latest/)

Therefore I made the following additions to the template.

### Detailed README instructions
The README.md file contains detailed instructions on how to use the tools included in the template once you have created a new project. The main sections are:
* [Prerequisites](https://github.com/VanderpoelLiam/cookiecutter-python/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/README.md#prerequisites)
* [Installation](https://github.com/VanderpoelLiam/cookiecutter-python/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/README.md#installation)
* [Usage](https://github.com/VanderpoelLiam/cookiecutter-python/blob/master/%7B%7Bcookiecutter.project_slug%7D%7D/README.md#usage)

### Configuration files
The `setup.cfg` file stores the configurations for flake8, isort and darglint. Any pytest configurations would also go here.

`setup.cfg`:
```
[flake8]
select = ANN,B,B9,BLK,C,D,DAR,E,F,I,S,W
ignore = E203,E501,W503
max-line-length = 80
max-complexity = 10
application-import-names = {{cookiecutter.package_name}},tests
import-order-style = google
docstring-convention = google
per-file-ignores = tests/*:S101

[isort]
profile = black

[darglint]
strictness = short
```

I use the default Python `.gitignore` provided by Github with the `pyenv` and `poetry` sections uncommented. 

### pre-commit hooks
The `pre-commit` file that runs before every commit to identify simple issues, lint and format the code.

`.pre-commit-config.yaml`:
```
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/psf/black
    rev: '23.3.0'
    hooks:
    -   id: black
-   repo: https://github.com/PyCQA/isort
    rev: '5.12.0'
    hooks:
    -   id: isort
-   repo: https://github.com/PyCQA/flake8
    rev: '6.0.0'
    hooks:
    -   id: flake8
        additional_dependencies: [
            "flake8-bandit",
            "flake8-black",
            "flake8-bugbear",
            "flake8-docstrings",
            "flake8-isort",
        ]
-   repo: https://github.com/terrencepreilly/darglint
    rev: 'v1.8.1'
    hooks:
    -   id: darglint
```

## Project creation
To create a new project with the default name `Foundations Of Psychohistory`, run:
```
❯ cookiecutter gh:/VanderpoelLiam/cookiecutter-python 
  [1/4] project_name (Foundations Of Psychohistory): 
  [2/4] project_slug (foundations-of-psychohistory): 
  [3/4] package_name (foundations_of_psychohistory): 
  [4/4] author (Hari Seldon): 

❯ cd foundations-of-psychohistory 

❯ git init
❯ git add . 
❯ git commit -m "Initial commit"

# Setup Python environment
❯ pyenv install 3.10.12 
❯ pyenv virtualenv 3.10.12 foundations_of_psychohistory
❯ pyenv local foundations_of_psychohistory 

# Setup Poetry
❯ poetry init --dev-dependency pre-commit --dev-dependency pytest --python 3.10.12
... follow the instructions ...

# Add sample project dependencies
❯ poetry add pandas
❯ poetry add --group dev pytest-cov

# Install the project
❯ poetry install

# Install pre-commit
❯ poetry run pre-commit install
```
This installation assumes you have the [prerequisites installed](#first-steps). 

This will result in the following directory structure:
```
foundations-of-psychohistory
├── .gitignore
├── poetry.lock
├── .pre-commit-config.yaml
├── pyproject.toml
├── .python-version
├── README.md
├── setup.cfg
├── src
│   └── foundations_of_psychohistory
│       └── __init__.py
└── tests
    └── __init__.py
```

## Conclusion
We have created the [cookiecutter-python](https://github.com/VanderpoelLiam/cookiecutter-python) template. To create a new project, run 
```
cookiecutter gh:/VanderpoelLiam/cookiecutter-python
```
then follow the instructions in the README of the new project.


## References
### Cookiecutters 
1. [Cookiecutter PyPackage fork](https://github.com/briggySmalls/cookiecutter-pypackage/tree/master)
2. [Hypermodern Python Cookiecutter Template](https://github.com/cjolowicz/cookiecutter-hypermodern-python)
3. [Cookiecutter Tutorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#)

### Project structure
1. [Hypermodern Python Blog](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
2. [Hypermodern Python Repository](https://github.com/cjolowicz/hypermodern-python)