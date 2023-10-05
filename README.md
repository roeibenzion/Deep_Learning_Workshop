# Deep_Learning_Workshop

This repository contains code for the Deep Learning Workshop projects, focusing on Floods, Heat Waves, and Storms. Follow the instructions below to set up the environment and run the tests for each project.

## Installation

### Prerequisites

- Ensure [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed on your system.
- Python is required. The version of Python used for the project will be installed in the Conda environment as described below.

### Clone the Repository

1. Open a terminal or command prompt.
2. Clone the repository and navigate into it:

    ```bash
    git clone https://github.com/roeibenzion/Deep_Learning_Workshop.git
    cd Deep_Learning_Workshop
    ```

### Set Up Conda Environment

Create and activate a new Conda environment using the provided `environment.yml` file:

1. To create an environment from the `environment.yml` file, run:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the newly created environment:

    ```bash
    conda activate env
    ```

*Note: Replace `env` with the actual name of your environment as specified in the `environment.yml` file.*

## Testing

Below are instructions for testing the three projects: Floods, Heat Waves, and Storms.

### Running Tests

Execute the `test.py` script with one of the following arguments to test the respective projects: `floods`, `heat_waves`, or `storms`.

Example:

```bash
python test.py floods      # for testing Floods
python test.py heat_waves  # for testing Heat Waves
python test.py storms      # for testing Storms
