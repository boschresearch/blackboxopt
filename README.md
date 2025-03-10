# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/boschresearch/blackboxopt/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                    |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| blackboxopt/\_\_init\_\_.py                             |        7 |        0 |        0 |        0 |    100% |           |
| blackboxopt/base.py                                     |       76 |        0 |       14 |        0 |    100% |           |
| blackboxopt/evaluation.py                               |       58 |        4 |        8 |        2 |     91% |88, 94, 101, 104 |
| blackboxopt/examples/\_\_init\_\_.py                    |        0 |        0 |        0 |        0 |    100% |           |
| blackboxopt/examples/dask\_distributed.py               |       17 |       17 |        2 |        0 |      0% |      6-44 |
| blackboxopt/examples/multi\_objective\_multi\_param.py  |       37 |        1 |        2 |        1 |     95% |        90 |
| blackboxopt/io.py                                       |       36 |        0 |        8 |        0 |    100% |           |
| blackboxopt/logger.py                                   |        3 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimization\_loops/\_\_init\_\_.py         |        0 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimization\_loops/dask\_distributed.py    |       75 |        5 |       16 |        0 |     95% |12-13, 96-97, 170 |
| blackboxopt/optimization\_loops/sequential.py           |       38 |        3 |        8 |        1 |     91% |68->71, 116-118 |
| blackboxopt/optimization\_loops/testing.py              |       31 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimization\_loops/utils.py                |       31 |        0 |        6 |        0 |    100% |           |
| blackboxopt/optimizers/\_\_init\_\_.py                  |        0 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimizers/bohb.py                          |       20 |        2 |        2 |        0 |     91% |     24-25 |
| blackboxopt/optimizers/botorch\_base.py                 |      120 |       18 |       28 |        7 |     82% |41-42, 67, 200->203, 215-239, 248, 279, 314-315, 318-323 |
| blackboxopt/optimizers/botorch\_utils.py                |       73 |        2 |       24 |        2 |     96% |   68, 208 |
| blackboxopt/optimizers/hyperband.py                     |       19 |        2 |        0 |        0 |     89% |     25-26 |
| blackboxopt/optimizers/random\_search.py                |       14 |        0 |        2 |        0 |    100% |           |
| blackboxopt/optimizers/space\_filling.py                |       15 |        2 |        0 |        0 |     87% |     10-11 |
| blackboxopt/optimizers/staged/\_\_init\_\_.py           |        0 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimizers/staged/bohb.py                   |      184 |       26 |       66 |        9 |     85% |69-70, 74-80, 131-132, 183->185, 266, 314, 337-354, 363-368, 380-384, 416, 451 |
| blackboxopt/optimizers/staged/configuration\_sampler.py |       15 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimizers/staged/hyperband.py              |       13 |        0 |        0 |        0 |    100% |           |
| blackboxopt/optimizers/staged/iteration.py              |       65 |        0 |       14 |        0 |    100% |           |
| blackboxopt/optimizers/staged/optimizer.py              |       44 |        0 |       10 |        0 |    100% |           |
| blackboxopt/optimizers/staged/utils.py                  |       26 |        3 |        8 |        3 |     82% |60, 66, 76 |
| blackboxopt/optimizers/testing.py                       |      161 |        4 |       52 |        5 |     96% |41->43, 46->48, 50, 139, 192, 335 |
| blackboxopt/utils.py                                    |       36 |        0 |       10 |        0 |    100% |           |
| blackboxopt/visualizations/\_\_init\_\_.py              |        0 |        0 |        0 |        0 |    100% |           |
| blackboxopt/visualizations/utils.py                     |       88 |       19 |       16 |        6 |     74% |51, 61-76, 137, 139, 142, 144, 204, 216 |
| blackboxopt/visualizations/visualizer.py                |      242 |       25 |       64 |        4 |     90% |54, 252->254, 270-272, 419-422, 637-686 |
|                                               **TOTAL** | **1544** |  **133** |  **360** |   **40** | **90%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/boschresearch/blackboxopt/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/boschresearch/blackboxopt/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/boschresearch/blackboxopt/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/boschresearch/blackboxopt/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fboschresearch%2Fblackboxopt%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/boschresearch/blackboxopt/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.