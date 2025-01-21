# RT_CETSA Dashboard (v0.4.0-dev0)

Web-based dashboard that manages a full RT_CETSA pipeline.

## Setup

The dashboard requires python >3.10.
It is recommended to create a dedicated environment for the project.

```conda create -n {env_name} python=3.10``` or ```python -m venv {env_name}```

## Install

Install all dependencies:

```pip install pyproject.toml```

or through [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```poetry install```

## Run the dashboard

```solara run src/rt_cetsa/dashboard.py```

## Data

All the data created will be stored in a `data` folder in the current working directory.