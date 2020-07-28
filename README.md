# Machine Ocean Work Package 1 and 2 Tools

Machine Ocean Website: [machineocean.met.no](https://machineocean.met.no/).

To use MOTools, you must add ot to your `PYTHONPATH`, for example add the following to your `.bashrc`:

```bash
export PYTHONPATH="${PYTHONPATH}:FILL_WITH_YOUR_PATH/MachineOcean-WP12"
```

Install dependencies with

```
pip3 install --user -r requirements.txt
```

Python3 package dependencies are as follows:

* tensorflow
* netcdf4
* sentinelsat

## Configuration

There are three config files used by this tool box:

* `main_config.json` is the main config file, and this file is provided with the repository.
* `met_config/met_config.json` is the main config file for MET internal use.
  This file is located in a submodule, and only accessible on MET Norway's network.
  The settings in this file overrides settings in the `main_config.json` file.
* `user_config.json` is the user's own config file with user-specific settings.
  Settings in the `main_config.json` and `met_config.json` file can be overridden here.
  This file is not provided, but a sample file is provided in the repo.
  Please copy and rename the `user_config_sample.json` file to `user_config.json`.

## Debugging

The default logging level of MOTools is `INFO`.
To change logging level, set the environment variable `MOTOOLS_LOGLEVEL` to another value.
Valid values are `CRITICAL`, `ERROR`, `WARNING`, `INFO`, and `DEBUG`.
See the Python logging [documentation](https://docs.python.org/3/library/logging.html#logging-levels) for more details.

For instance:
```bash
export MOTOOLS_LOGLEVEL=DEBUG
```
