---
title: "uv Python Package Manager"
date: 2025-04-03T15:57:12+02:00
draft: true
---

Astral have created an incredible tool [uv](https://docs.astral.sh/uv/) that replaces the frankly terrible system I have been using to manage python versions, projects and virtual environments. The idea is to go through an example of setting up a repository managed by uv using [cookiecutter-uv](https://github.com/VanderpoelLiam/cookiecutter-uv).

## Initial Setup

The prerequisite is that uv is installed, so follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/). Then run:

```shell
uvx cookiecutter https://github.com/VanderpoelLiam/cookiecutter-uv.git
```

and follow the prompts to setup the repo. Note: This is all explained in the [cookiecutter-uv README](https://github.com/VanderpoelLiam/cookiecutter-uv). I picked all the default choices which resulted in the repository:

```shell
> tree pg-doom/
pg-doom/
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── docs
│   ├── index.md
│   └── modules.md
├── mkdocs.yml
├── pyproject.toml
├── src
│   └── pg_doom
│       ├── __init__.py
│       └── foo.py
├── tests
│   └── test_foo.py
└── tox.ini
```

Then following the instructions in the newly created README, we first created the repository [pg-doom](https://github.com/VanderpoelLiam/pg-doom) on GitHub, then run:

```shell
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:VanderpoelLiam/pg-doom.git
git push -u origin main
```

It is normal that the GitHub Actions fail on this first push, that is because we need to create our lockfile. 

```shell
make install
uv run pre-commit run -a
```

Now when we push the changes everything should work:

```shell
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

We can then safely delete all the redundant info in the README, I also additionally moved the github actions code to deploy the docs from `.github/workflows/on-release-main.yml` to `.github/workflows/main.yml` as I do not intend to release this code as a package, so it makes sense to build the docs on each push to main. Enabling the documentation also requires navigating to `Settings > Actions > General` in the repository, and under `Workflow permissions` select `Read and write permissions` then click `Save`. Additionally navigate to `Settings > Pages > Build and deployment`, and under `Build and deployment` select `gh-pages`. The docs should then be built and available at [https://vanderpoelliam.github.io/pg-doom/](https://vanderpoelliam.github.io/pg-doom/) otherwise read [the docs from the forked cookiecutter](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
