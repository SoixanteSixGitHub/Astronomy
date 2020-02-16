# Astronomy Baseline
A baseline for Astronomy competition on biendata platform.

## Solution
We use a MLP to accomplish the classification task. Network architecture is defined in Model.py.

Three classes 'star', 'galaxy', 'qso' are mapped to labels 0, 1, 2 respectively.

## Code Orginization
* The network architeture is defined in Model.py

* The dataset class is defined in Dataset.py

* main function to train and validate is defined in main.py

* i also provide a utility function, namely macro_f1, to evaluate your result(a csv file) which is defined in utils.py.


## Folders
the name of folders should have shown their function.
- /data is folder to data 

- /checkpoint is folder for saving model weights

- /results is folder where result csv files are saved. (put val_labels_v1.csv here also!)

- /logs is folder you can check validate infos.

## Dependency
+ pytorch >= 1.0 
+ numpy 
+ pandas

\#note: lr_scheduler should be put in the end of each epoch if using pytorch>=1.1

## How to run
`python main.py`
