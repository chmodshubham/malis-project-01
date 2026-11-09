# MALIS Project 1: Fishing Season and Multi-Armed Bandits

Code for MALIS Project 1 by Soulaimen (log `bouazza`), Shubham Kumar (log `kumarsh`) and Adhitya (log `kanakara`). `agents.py` holds every agent, and `experiment.ipynb` runs every experiment of the report.

## Prerequisites

| Tool                                     | Version            | Check             |
| ---------------------------------------- | ------------------ | ----------------- |
| conda (Miniforge, Miniconda or Anaconda) | any recent version | `conda --version` |

The environment below installs Python 3.12, numpy, matplotlib and Jupyter. No other package is needed. Installing conda and the packages needs an internet connection.

### Install conda

Skip this step if `conda --version` already prints a version. The commands below install Miniforge, a free and small conda distribution, in your home folder.

On macOS or Linux, in a terminal:

```bash
cd ~

# Download the Miniforge installer for this computer
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Install it in ~/miniforge3 without asking questions
bash "Miniforge3-$(uname)-$(uname -m).sh" -b

# Make the conda command available in new terminals
~/miniforge3/bin/conda init "$(basename "$SHELL")"

# Delete the installer, it is no longer needed
rm "Miniforge3-$(uname)-$(uname -m).sh"
```

Close the terminal and open a new one. `conda --version` must now print a version.

On Windows, in Command Prompt (`cmd.exe`):

```bat
cd %UserProfile%

rem Download the Miniforge installer
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

rem Install it in %UserProfile%\Miniforge3 without asking questions
start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniforge3

rem Delete the installer, it is no longer needed
del Miniforge3-Windows-x86_64.exe

rem Make the conda command available in this window
%UserProfile%\Miniforge3\Scripts\activate.bat
```

Run the commands of the next sections in the same window. In a new Command Prompt window, run the last command above again first.

> [!NOTE]
> In Command Prompt, use `cd /d <submission folder>` instead of `cd <submission folder>` when the folder is on another drive, for example `D:`.

## Setup

Unzip the submission, then run these commands. Replace `<submission folder>` with the folder that holds `agents.py`.

```bash
cd <submission folder>

# Create the environment
conda create -n malis python=3.12 numpy matplotlib jupyter -y

# Turn the environment on
conda activate malis
```

## Running the experiments

```bash
cd <submission folder>

# Turn the environment on
conda activate malis

# Run the notebook from top to bottom and save its outputs
jupyter nbconvert --to notebook --execute experiment.ipynb --output experiment.ipynb
```

The run takes about 50 seconds. It also writes the report figures to `figures/`.

To work interactively instead, open the notebook in the browser, then use Kernel, Restart and Run All:

```bash
cd <submission folder>

# Turn the environment on
conda activate malis

# Open the notebook in the browser
jupyter notebook experiment.ipynb
```

## Configuration

The notebook expects these files next to it:

| Path                             | Content                   |
| -------------------------------- | ------------------------- |
| `fishing.py`                     | The provided simulator    |
| `agents.py`                      | All agents                |
| `logs/fishing_log_bouazza.json`  | Game log of Soulaimen     |
| `logs/fishing_log_kumarsh.json`  | Game log of Shubham Kumar |
| `logs/fishing_log_kanakara.json` | Game log of Adhitya       |

Seeds are fixed in the notebook (`TUNE_SEEDS`, `TEST_SEEDS`, `CONTEXT_SEEDS`), so every run gives the same results.

## Cleanup

```bash
# Turn the environment off
conda deactivate

# Delete the environment
conda env remove -n malis -y
```

If you installed Miniforge with the steps above and no longer need it, remove it too. On macOS or Linux:

```bash
# Undo the conda setup of your shell
conda init --reverse "$(basename "$SHELL")"

# Delete Miniforge and its settings
rm -rf ~/miniforge3 ~/.conda ~/.condarc
```

> [!WARNING]
> `rm -rf` cannot be undone. Run it only if `~/miniforge3` is the Miniforge you installed for this project.

On Windows, open Settings, then Apps, find Miniforge3 in the list of installed apps, and uninstall it.
