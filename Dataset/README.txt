====================================
DDTI dataset:
    The DDTI dataset is downloaded from the official website: http://cimalab.intec.co/applications/thyroid/
    - The labels are in ../DDTI/label.txt
    - The ultrasound images are the jpg files: ../DDTI/*.a.jpg
    - The nodule masks are: ../DDTI/*.b.jpg
    All the ultrasound images are the original images from the official website.
    The masks are drawn according to the numeric boundary in the official xml files.
    The labels are gotten from the official xml file as well.
    NB: we downloaded the images one by one from the website using the individual links because the links to download all cases sometimes misses images.

    In label.txt, each line corresponds to an image. The columns are: 
    #   0: image id (ultrasound image file name)
    #   1: folder names obtained from the 'Download database' on the official website -- 'bening', 'maling', 'na'
    #      NB: almost all the previous works (including this work) ignore this label, but use '6:tirads' as the bening/maling label.
    #   2: composition -- 'solid', 'predominantly-solid', 'predominantly-cystic', 'cystic', 'spongiform', ''
    #   3: echogenicity -- 'marked-hypoechogenicity', 'hypoechogenicity', 'isoechogenicity', 'hyperechogenicity', ''
    #   4: margins -- 'well-defined', 'microlobulated', 'macrolobulated', 'ill-defined', 'spiculated', ''
    #   5: calcifications -- 'non', 'microcalcification', 'macrocalcification', '' 
    #   6: tirads -- '2', '3', '4a', '4b', '4c', '5', ''
    #      NB: usually regard '2', '3' as 'bening'; '4a', '4b', '4c', '5' as 'maling'
    #   7: sex -- 'F', 'M', ''
    #   8: age

    @inproceedings{pedraza2015open,
        title={An open access thyroid ultrasound image database},
        author={Pedraza, Lina and Vargas, Carlos and Narv{\'a}ez, Fabi{\'a}n and Dur{\'a}n, Oscar and Mu{\~n}oz, Emma and Romero, Eduardo},
        booktitle={10th International Symposium on Medical Information Processing and Analysis},
        volume={9287},
        pages={92870W},
        year={2015},
        organization={International Society for Optics and Photonics}
    }


====================================
BUSI dataset:
    There are two versions of the BUSI dataset.
    Version 1. from: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
    Version 2. from: https://www.kaggle.com/anaselmasry/datasetbusiwithgt
    The differences are:
    -- Version 1 contains the ultrasound images with nodule masks, but Version 2 only contains ultrasound images.
    -- Version 1 does not split the datasets to training/testing sets, but Version 2 does.
    We downloaded the Version 1.
    We used the masks for baseline training, but our model does not use the masks.

    @article{al2020dataset,
        title={Dataset of breast ultrasound images},
        author={Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
        journal={Data in Brief},
        volume={28},
        year={2020},
        publisher={Elsevier}
    }