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

## What features to include in the template?
I am basing my template mainly off the [Hypermodern Python Cookiecutter Template](https://github.com/cjolowicz/cookiecutter-hypermodern-python). However I only want the following features:
* Packaging and dependency management with [Poetry](https://python-poetry.org/)
* Linting with [pre-commit](https://pre-commit.com/) and [Flake8](https://flake8.pycqa.org/en/latest/)
* Code formatting with [Black](https://github.com/psf/black)
* Import sorting with [isort](https://pycqa.github.io/isort/)
* Testing with [pytest](https://docs.pytest.org/en/latest/)

### Requirements
Update the `README.md` with the installation requirements to create a new project:
````
## Requirements
(Reccomended) Install [pyenv](https://github.com/pyenv/pyenv) to manage Python versions:
```
curl https://pyenv.run | bash
```

Install [pipx](https://github.com/pypa/pipx):
```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

Install [Cookiecutter](https://github.com/cookiecutter/cookiecutter):
```
pipx install cookiecutter
```

Install [Poetry](https://python-poetry.org/)
```
pipx install poetry
```
````

### Configuration files
Create the `setup.cfg` file that stores the configurations for flake8, isort and darglint. Any pytest configurations would also go here.

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

I use the default Python `.gitignore` provided by Github. 

### pre-commit hooks
Create the `pre-commit` file that runs before every commit to identify simple issues, lint and format the code.

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

### Development environment
Update the `README.md` with the development environment setup instructions.
````
## Development environment (reccomended)
We reccomend this approach for managing the development environment.

Install Python 3.10.12:
```
pyenv install 3.10.12
```

Create a new virtualenv:
```
pyenv virtualenv 3.10.12 {{cookiecutter.package_name}}
```

Set this to the default Python environment inside the repository:
```
pyenv local {{cookiecutter.package_name}}
```
````

### Project creation
Update the `README.md` with the project creation instructions. For clarity, I provide the below instructions with the fields filled in i.e. instead of {{cookiecutter.project_name}} I fill in `Foundations Of Psychohistory`.
````
## Create a new project
Example usage:
```
❯ cookiecutter gh:/VanderpoelLiam/cookiecutter-python 
  [1/4] project_name (Foundations Of Psychohistory): Foundations Of Psychohistory
  [2/4] project_slug (foundations-of-psychohistory): foundations-of-psychohistory
  [3/4] package_name (foundations_of_psychohistory): foundations_of_psychohistory
  [4/4] author (Hari Seldon): Hari Seldon

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
````

## Final structure
Following the sample instructions in [project creation section](#project-creation) will result in the following directory structure:
TODO: incorrect directiory structure
```
{{cookiecutter.project_slug}}
├── README.md
├── src
│   └── {{cookiecutter.package_name}}
│       └── __init__.py
├── tests
│  └── __init__.py
│
└── setup.cfg
```

# TODO:
* Link the the accompanying cookie cutter template I create
* Fix issue with .precommit-config.yaml not being copied over

## References
### Cookiecutters 
1. [Cookiecutter PyPackage fork](https://github.com/briggySmalls/cookiecutter-pypackage/tree/master)
2. [Hypermodern Python Cookiecutter Template](https://github.com/cjolowicz/cookiecutter-hypermodern-python)
3. [Cookiecutter Tutorial](https://cookiecutter.readthedocs.io/en/stable/tutorials/tutorial2.html#)

### Project structure
1. [Hypermodern Python Blog](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
2. [Hypermodern Python Repository](https://github.com/cjolowicz/hypermodern-python)