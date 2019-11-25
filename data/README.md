# Dataset
You could download the dataset from https://drive.google.com/file/d/1fThys1-OKgsDcy0wTRhf0eYU4zSrB7dm/view?usp=sharing. 

The dataset contains four files, including train.txt, valid.txt, test.txt and relation2id.txt.

The structures of train.txt, valid.txt, test.txt are homogeneous. Each line in those files is a instance, containing a entity pair and a corresponding sentence. The first element in each line is the id of the head entity, and the second element is the id of the tail entity. The subsequent two elements are the name of head and tail entity respectively, and the fifth element is the relation type. The later part in each line is a sentence corresponding with this entity pair, ended by a special token ‘###END###’.

The relation2id.txt records the name of all relation types, each name resides in a line.

More statistics and details of this dataset are shown in the paper.  

An additional file of pre-trained word emebdding, vec4.bin, is also attached in the dataset.
