README
------

Submitters:
-----------
Ori Braverman 318917010
Elie Nedjar 336140116

Project Description
-------------------
The project is divided into 5 parts, each part is a different model for tagging words in a sentence.
The models are as follows:
1. A simple window-based tagger
2. External word embeddings
3. Adding the external word embeddings to the tagger
4. Adding sub-word units
5. Convolution-based sub-word units

Project Structure
-----------------
assignment_2
├── Data
│   ├── ner
│   │   ├── ._dev
│   │   ├── ._train
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── pos
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── vocab.txt
│   └── wordVectors.txt
├── Part_1
│   ├── Output
│   │   ├── test1.ner
│   │   └── test1.pos
│   ├── Part1.pdf
│   └── tagger1.py
├── Part_2
│   ├── Part2.pdf
│   └── top_k.py
├── Part_3
│   ├── Output
│   │   ├── test3.ner
│   │   └── test3.pos
│   ├── Part3.pdf
│   └── tagger2.py
├── Part_4
│   ├── Output
│   │   ├── part4.ner
│   │   └── part4.pos
│   ├── Part_4.pdf
│   └── tagger3.py
├── Part_5
│   ├── Output
│   │   ├── test5.pos
│   │   └── test5.ner
│   ├── Part5.pdf
│   └── tagger4.py
├── README.md
├── README.txt
├── ass2.pdf
└── utils.py

How to run the code
-------------------
1. Make sure you have the required libraries installed. You can install them by running the following command:
   pip install -r requirements.txt
2. Run the following command to run the code:
    python3 tagger1.py
    python3 top_k.py
    python3 tagger2.py
    python3 tagger3.py
    python3 tagger4.py

    Note: Make sure the working directory is Assignment_2, otherwise the code will not fetch the data correctly.
    Also, inside each python file, you can change the parameters of the model, such as pos or ner, the number of epochs, etc.

