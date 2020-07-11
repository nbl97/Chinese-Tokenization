# Chinese-Tokenization
Authored by Chen Z.Y. and Ni B.L..

## Introduction
This is a project to handle with Chinese Tokenization. It is also our homework for course of Natural Language Processing.


We've implemented the followings
- Rule-based tokenization with a shortest path.
- Simple 2-gram language model without part-of-speech
- HMM model with part-of-speech
- Character-based tagging

## Installation
We develop with Python3.7 on Windows 10. However, we also run our codes successfully on Linux.

The only special package we use is `ahocorasick`. Run the following command to install. Please note that it requires C++ lib. Install `Microsoft Visual C++ Build Tools` on Windows.

`pip install pyahocorasick`

`git clone https://github.com/volgachen/Chinese-Tokenization`

## Examples

` python test_exp1.py --use-re --score Markov` to experiment with rule-based algorithms. You can remove `--use-re` if you do not want re-replacement. `--score` can be choosen from None, Markov and HMM, which decides if we combine with language models.

` python test_exp2.py ` to see results with simple 2-gram lauguage model. Several results with different datasets will be shown. You can run `python test_exp2.py train` or `python test_exp2.py test` to execute evaluation within/without training set.

` python BMES_exps/BMES.py ` to run experiments about character-based tagging. We could see experiments with different train/val set and experiments with or without re-replacement. You need first run `python BMES_exps/convert_BMES.py` to get organized corpus for this experiment.

