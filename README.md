# ANLP-Assignment-2
## Translational Model
 The directory structure of submission is as follows

```
.
├── ANLP_Assignment_2.pdf
├── HyperParameter Tuning
│   ├── Combination-1
│   │   ├── main.ipynb
│   │   ├── model.pth
│   │   ├── plot.png
│   │   └── testbleu.txt
│   ├── Combination-2
│   │   ├── main .ipynb
│   │   ├── model .pth
│   │   ├── plot.png
│   │   └── testbleu.txt
│   ├── Combination-3
│   │   ├── main.ipynb
│   │   ├── plot.png
│   │   └── testbleu.txt
│   ├── Combination-4
│   │   ├── main.ipynb
│   │   ├── model.pth
│   │   ├── plot.png
│   │   └── testbleu.txt
│   └── Combination-5
│       ├── main.ipynb
│       ├── model.pth
│       ├── plot.png
│       └── testbleu.txt
├── README.md
├── Report.pdf
├── src
│   ├── decoder.py
│   ├── encoder.py
│   ├── misc
│   │   ├── hyperparameters.pkl
│   │   └── vocab.pkl
│   ├── model.py
│   ├── __pycache__
│   │   ├── decoder.cpython-311.pyc
│   │   ├── encoder.cpython-311.pyc
│   │   ├── model.cpython-311.pyc
│   │   ├── train.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   ├── testbleu.txt
│   ├── test.py
│   ├── train.py
│   ├── transformer.pt
│   └── utils.py
├── ted-talks-corpus
│   ├── dev.en
│   ├── dev.fr
│   ├── test.en
│   ├── test.fr
│   ├── train.en
│   └── train.fr
└── Transformer.ipynb

```

Few pointers on above directory structure:
- The **ted-talks-corpus** includes dataset files.
- The final python script are in src folder in addition to `Transformer.py` for complete code in python notebook.
- The **HyperParameter Tuning** includes python notebook with saved outputs to different combinations along with loss graph and bleu score corresponding to each sentence in testing.

Instructions on running the script:
- Goto the **src** folder.
- Run `python3 train.py` for training
- Run `python3 test.py` for testing

Assumptions:
- Introduction of `model.py` for better modularity
- Ignoring `<pad>` token while calculating loss
- I have omitted the punctuation for training and testing, though it gave better results with it.
- I have replaced words with frequency less than certain threshold wiht `<unk>` tokken for better training and testing.

