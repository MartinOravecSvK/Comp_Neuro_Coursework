# Computational Neuroscience Coursework COMS30017 (2023/24)

### Mark: 87% (Marked by Dr. Conor Houghton)

Notes from the marker:

Nice graphs apart from the tiny caption, resize in code not in the document! Great answer to question 3 with interesting graphs, very innovative; Q6 was also interesting though not a complete. Well done.

*Q1-9* **|** *Q2-9* **|** *Q3-15* **|** *Q4-8* **|** *Q5-8* **|** *Q6-12*

61/70

## Report

The report was written in LaTeX and both it's compiled and code version is [here](./report/) and here is the report pdf directly [report](./report/Computational_Neuroscience_Coursework.pdf).

Also [here](./ExtendedCoursework/) are all the files given for the coursework (questions, data).

Notes:

Please keep in mind this work has been done in a coursework period which is 3 weeks to do 2 coursework assignments so the total amount of work was realistically less than 1.5 weeks!

There have been minor changes to the repository since just to make it a bit nicer as a public repository. The underlying code has not been touched apart from some comments and a little bit of cleaning of unused code.

## Introduction

The coursework consisted of 6 questions and questions 3 and 6 were open-ended, meaning we (as students) had the freedom to experiment with what we thought was a good idea.

The marking scheme reflected this as questions 1, 2, 4 and 5 were 10 points each and questions 3 and 6 were 15 each respectively.
Bit more on the marking scheme 70%+ ~ A in the US system and 90%+ publish worthy work.

The code is largely divided based on a question it is intended to solve but may be used somewhere else as a helper function and I didn't bother putting it in the utils, which contain some of the functions used throughout the code.

Some of the code takes longer to execute, i.e. several hours depending on the CPU and simulation parameters. Longer simulations and more simulations (for averaging) will obviously increase the time. The code does scale linearly with the number of CPU threads, excluding some overhead.

<!-- Finish this part -->
## Technology used:
Python, MatplotLib, Seaborn, 

## Running the code

 - First clone the repository with HTTPS:

    ```bash
    git clone https://github.com/MartinOravecSvK/Comp_Neuro_Coursework.git
    ```
    or via ssh.
    ```bash
    git clone git@github.com:MartinOravecSvK/Comp_Neuro_Coursework.git
    ```
 - Load conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate sim_env
    ```

### Running solutions for questions 1, 2, 4, 5

These are simple and quick computationas. Each has its own respective file but are also executed together in the main.

To run main:

```bash
python main.py
```

Or to run specific script for a question:

```bash
cd src/
python qX.py
```
Replacing the X with 1, 2, 4 or 5.

To run q3q6 file you should understand basic simulation parameters as the simulation itself could take a very long time to execute.
