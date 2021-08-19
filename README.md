# SRS_KAeDCN
This is the sourcecode for the paper: "A Knowledge-Aware Recommender with Attention-Enhanced Dynamic Convolutional Network".

## Citation:
```
@inproceedings{Liu2021,
  title={A Knowledge-Aware Recommender with Attention-Enhanced Dynamic Convolutional Network},
  author={Liu, Yi and Li, Bo-Han and Zang, Yalei and Li, Aoran and Yin, Hongzhi},
  booktitle={CIKM},
  year={2021}
}
```

## Requirements:
* Python 3
* Numpy
* Pytorch v1.9

## Configurations
### Data:
    In file: all_data, there are 4 datasets from 3 resources: ml-1m, ml-20m, book and music. Each sub_file has the RS data and KG embedding data separately named ***.txt and ***.npy

## Usage

1.  Select the dataset you want. 
2.  Put the RS data in file: Data and the KG data in file: KG_data.
3.  Reset the parameter: "dataset" in file: <code>main.py</code> as the name of RS data you choose.
4.  Run <code>python main.py</code>



## Acknowledgement

1.  This code is based on [DynamicRec](https://github.com/Mehrab-Tanjim/DynamicRec). 
2.  Our datasets are supported by [the works of Jin Huang, et al](https://github.com/RUCAIBox/KB4Rec).

    Thank them for their works! 
    
    If you find the content useful, we also suggest you cite their works:
    ```
    @inproceedings{tanjim2020dynamicrec,
    title={DynamicRec: A Dynamic Convolutio nal Network for Next Item Recommendation.},
    author={Tanjim, Md Mehrab and Ayyubi, Hammad A and Cottrell, Garrison W},
    booktitle={CIKM},
    pages={2237--2240},
    year={2020}
    }
    @inproceedings{KSR-SIGIR-2018,
    author={Jin Huang and Wayne Xin Zhao and Hongjian Dou and Ji{-}Rong Wen and Edward Y. Chang},
    title={Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks},
    booktitle = {The 41st International {ACM} {SIGIR} Conference on Research {\&} Development in Information Retrieval, {SIGIR} 2018, Ann Arbor, MI, USA, July 08-12, 2018},
    pages     = {505--514},
    year      = {2018},
    }
    ```
    All the content is only for the scholarly communication. For infringement, please contact us deleted.
