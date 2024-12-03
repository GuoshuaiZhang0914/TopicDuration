## Introduction
In this project, we used history chat content of three different alliances obtained directly from strategy game Nova Empire II to reveal the topic duration via factors analysis of participator, content, and social network for social system applications.

## Requirements
Install dependency in a Python>=3.7.0 environment

    pip install bertopic
    pip install nltk
    pip install pandas
    pip install networkx


## Instruction
1. "topic_extract.py" is used for extracting topics from each chat content by BERTopic.

2. "topic_tracking_plot.py" is used for obtaining multiple topics tracklet by Hungarian algorithm, and plotting curves for each dynamic topic. 

3. "topic_tracking_properties.py" is used for computing the 20 potential factors for each topic.

4. Factor analysis, regression analysis, and correlation analysis are performed by IBM SPSS Statistics 25.

## Dataset
Due to the data security, privacy protection and intellectual property rights, we cannot share the data used in this project in public. If you really need it, you can contact us.

## Citation
If you use this project in your research, please cite our work by using the following BibTeX entry:
```
@article{zhang2024modeling,
  author={Zhang, Guoshuai and Wu, Jiaji and Jeon, Gwanggil and Wang, Penghui and Chen, Yuan and Wang, Yuhui and Tan, Mingzhou},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={Modeling the Contributions of Participator, Content, and Network to Topic Duration in Online Social Group}, 
  year={2024},
  volume={11},
  number={6},
  pages={7146--7158},
  doi={10.1109/TCSS.2024.3414586}
}
```

## Contact
Be sure to let us know when you use the project for academic researches and commercial purposes.

Guoshuai Zhang, School of Electronic Engineering, Xidian University (E-mail: zhangguoshuai@xidian.edu.cn)


Copyright © 2024, Guoshuai Zhang. All Rights Reserved.
