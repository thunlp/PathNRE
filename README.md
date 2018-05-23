Modified on my own linux
# PathNRE
Codes and Dataset for EMNLP2017 paper ‘‘Incorporating Relation Paths in Neural Relation Extraction’’.

## Cite

If you use the codes or dataset, please cite the following paper:

[Zeng et al., 2017] Wenyuan Zeng, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Incorporating Relation Paths in Neural Relation Extraction. In Proceedings of EMNLP.

## Codes

The source codes of baselines and our methods are put in the folders CNN+rand/, CNN+max/, Path+rand/, Path+max/ respectively.

## Dataset

You could find the download link and description of the dataset from data folder. To run the model, you need to download and unzip the dataset, and put it into the folder of this repository.

## Train

For training, you need to type the following command in each model folder:

```bash
g++ train_cnn.cpp -o Train -O3 -pthread
./Train
```

The training model file will be saved in folder ./out/ .

## Test

For testing, you need to type the following command in each model folder:

```bash
g++ work.cpp -o Test -O3 -pthread
./Test
```

The testing result which reports the precision/recall curve will be shown in ./out/pr.txt.

