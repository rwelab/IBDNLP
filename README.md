## UCSF Medical Note Pipeline for Identifying Inflammatory Bowel Disease

In this project, our goal is to establish a rules-based approach for identifying important features present in select UCSF medical notes pertaining to Inflammatory Bowel Disease (IBD), with the aim of labeling each note with three specific symptoms related to IBD: abdominal pain, diarrhea, and blood in stool (also referred to as fecal blood).

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Diagrams](#diagrams)
- [Contributing](#contributing)
- [License](#license)

## Description


<img src="diagrams/abdominal_tree.jpg" alt="Example of Annotation Guidelines" width="400"/>

In the medical field, we have taken note of two points: 1) we have seen that the simple diagnosis of patients via noted symptoms can sometimes be difficult, but even more so if the decision is made by notes from a patient visit, rather than the visit itself, and 2) while most attention related to the notes of doctors pertains to their handwriting, our research humbly suggests that even after their scrawl is deciphered, their grammar is equally as cryptic. Thus, the goal of this pipeline attempts to address these two points by A) constructing a pipeline that can parse these notes according to the factors identified by our resident gastroenterologists and form them into a tabular format, and B) using these newly constructed features to create models for attaching labels to these notes regarding the three main symptoms mentioned earlier: abdominal pain, diarrhea, and fecal blood.

## Getting Started

Please begin by downloading the git repository. This can be done via download, or with the following line:

```
git clone https://github.com/rwelab/IBDNLP.git
```


### Prerequisites

This code was established using conda version 4.13.0, thus we recommend having this as the minimum conda version. Apart from this, all relevant packages needed to run this code have been included in the ```environment.yml``` file.

### Installation

To begin, we highly recommend establishing a conda environment using the provided yml file:

```
conda env create -f environment.yml -n ENVIRON_NAME
```

This will create the environment with all relevant packages that were used to construct our pipeline. From there, we must also download two SpaCy models that will be used for tokenization and word vector comparison.

```
python install_spacy_models.py
```


## Usage

Once your environment has been set up, and all files have been downloaded, please reference the ```user_definition.py``` file and change the column names to suit your use case. For reference, the following columns were used to develop and train our models:
 - "note": the text of the note itself
 - "deid_note_id": id for each note
 - "deid_PatientDurableKey": key used to identify patients
 - "deid_service_date_cdw": service date for the patient's visit/medical appointment

This is also where you can specify the location of your ```data``` folder.

Once this is done, the augmented data set can be created with the following steps:
 - 1) Create a ```data``` folder if not already done so
 - 2) Run this line in the ```pipe``` subfolder: 
 ```
 python data_extraction.py YOUR_DATA_FILE_PATH
 ```
 - 3) Wait for the pipeline to complete