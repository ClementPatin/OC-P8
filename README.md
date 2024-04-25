# OC-P8
# Semantic Segmentation

## Projet scenario


Future Vision Transport is a fictitious company that designs embedded computer vision systems for autonomous vehicles.

The project presented here focuses on the image segmentation part, using Cityscpaes dataset :
- build a data generator
- use data augmantation 
- compare some solutions
- build a first semantic segmentation model
- MLOps (deploy API, deploy UI, CI/CD with github actions)
- write a technical note and a presentation


## organization of the github repository

```bash
│   .gitignore
│   Patin_Clement_1_scripts_042024.ipynb
│   Patin_Clement_4_note_technique_042024.pdf
│   Patin_Clement_5_presentation_042024.pdf
│   README.md
│   myFunctions.py
│   requirements.txt
├───.github
│   └───workflows
├───.ipynb_checkpoints
├───mySaves
│   ├───models
│   ├───other_images
│   └───training_times
├───Patin_Clement_2_API_042024
│   ├───.dockerignore
│   ├───Dockerfile
│   ├───main.py
│   ├───requirements.txt
│   ├───TfLite
│   └───__pycache__
├───Patin_Clement_3_application_Flask_042024
│   ├───.dockerignore
│   ├───Dockerfile
│   ├───app.py
│   ├───requirements.txt
│   ├───static
│   │   ├───css
│   │   ├───predicted_mask
│   │   ├───test_images
│   │   └───test_masks
│   ├───templates
│   └───__pycache__
└───__pycache__

```


## other needed files and folders in local

The repo needs other files and folders to run :

```bash

├───P8_Cityscapes_gtFine_trainvaltest
├───P8_Cityscapes_leftImg8bit_trainvaltest
│
```