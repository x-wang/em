# Elastic consensus method for matching recallfixations onto encoding fixations
Match recall fixations to encoding using elastic matching algorithm

This repository contains the source code and dataset of the artilce "A consensus-based elastic matching algorithm for mappingrecall fixations onto encoding fixations in the looking-at-nothingparadigm". More information can be found on the [Project page](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/em.html).

## Setup 
- Install dependencies using pip/anaconda/etc, if needed: numpy, scipy, matplotlib, joblib, pillow ...
- Download the stimuli set from [link](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/dataset/images.zip) and unzip in main folder
- Download the eye movement data from link and unzip in [link](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/dataset/imgData.zip) the data folder

File structure:
```
em
|-- data
|   |-- imgData
|-- stimuli
|   |-- *.jpeg
|-- src
|-- res
```

## To run:

```
cd src/scripts/
python em.py
```

The recommended parameter settings are included in the source file ```em.py``` as well. Note that these parameters might depend on some experimental parameters, and should be adapted correspondingly. 

### Main function calls

- ```mapRecallToView``` in ```em.py``` computes the elastic mapping function given a set of recall fixations and the corresponding encoding fixations. 
- ```find_neighbors_within_radius``` in ```em.py``` filters encoding fixations based on recolated recall fixations. 
