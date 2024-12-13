import os

def get_dataset_path(name):
    if name == 'ChangeSim':
        """
        ├── Query
        │   ├── Query_Seq_Test
        │   └── Query_Seq_Train
        └── Reference
            ├── Ref_Seq_Test
            └── Ref_Seq_Train
        """
        return './data/ChangeSim'

    if name == 'CMU_binary':
        """
        ├── test
        │   ├── mask
        │   ├── t0
        │   └── t1
        └── train
            ├── mask
            ├── t0
            └── t1
        """
        return './data/VL-CMU-CD-binary255'
    
    if name == 'PCD_raw':
        """
        ├── GSV
        │   ├── mask
        │   │   └── *.png
        │   ├── README.txt
        │   ├── t0
        │   │   └── *.jpg
        │   └── t1
        │       └── *.jpg
        └── TSUNAMI
            ├── mask
            │   └── *.png
            ├── README.txt
            ├── t0
            │   └── *.jpg
            └── t1
                └── *.jpg
        """
        return './data/PCD'
