# batect-ml-template

A template for ML projects, with dependency management made effortless by `batect`

## Prerequisites

- [Docker](https://docs.docker.com/desktop/)
- Java 8 or newer (used by batect. you'll not be seeing Java code - I promise)
- On Linux and macOS: `bash` and `curl`
- On Windows: Windows 10 / Windows Server 2016 or later

## Setup

Install the dependencies needed by `batect` and your IDE on your host machine

```shell script
# mac users
bin/non_batect/go.sh

# windows / linux
# work in progress. in the meantime, please install Docker and Java manually if it's not already installed
```

Configure your IDE to use the python virtual environment (`./.venv/`) created by `go.sh` 
- [PyCharm instructions](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#existing-environment)
- [VS Code instructions](https://code.visualstudio.com/docs/python/environments)

## Tasks that you can run

```shell script
# run unit tests
./batect unit_test

# train ML model
./batect train_model
```