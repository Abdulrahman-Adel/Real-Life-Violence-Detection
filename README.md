
# Real-Life Violence Detection


Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_data_loader.py   - this file contains the abstract class of the data loader.
│   └── base_trainer.py   - this file contains the abstract class of the trainer.
│
│
├── models              - this folder contains the models for the project.
│   └── model_01.py
│
│
├── trainers             - this folder contains trainers of your project.
│   └── trainer.py
│   
│  
├──  data _loader  
│    └── data_loader_01.py  - data loader responsible for handling data generation and preprocessing
│
│
├── train.py --  main used to run the training across different config files and models
│
├── evaluate.py --  files responsible for the evaluation of different models. Loading and selecting the best model
│ 
├── train_bash.sh --  example bash script to run the training with different arguments
│
└── utils
     ├── dirs.py
     └── factory.py
     └── config.py
     └── utils.py


```

# Configuration
Config files are used to choose the models, trainers, and dataloader files for each project. so multiple project can co-exist in this template.
Check the config file contents for more info.


 