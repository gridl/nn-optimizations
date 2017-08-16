# nn-optimizations

This repo should contain various methods of the neural networks optimization approaches.

Desired structure:
```
nn-optimization/
├── optimization_type/
│   ├── __init__.py
│   ├── model.py
│   └── trainer.py
├── another_optimization/
│   ├── __init__.py
│   ├── model.py
│   └── trainer.py
├── data_downloader.py
├── README.md
├── requirements.txt
└── setup.py
```

We assume that models are written with pytorch framework([docs](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/))

# Installation

Sometimes it may be necessary to create import like `from optimization_type.model import Model`. To resolve such issue I propose to perform such steps:

- Install pytorch from [official site](http://pytorch.org/)
- Install all other required packages with `pip install -r requirements.txt`. I highly advice perform this step inside virtual environment.
- Install current repo in edit mode with `pip install -e .`
- During models creation add `__init__.py` file inside the folder
