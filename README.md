# Elastic consensus method for matching recallfixations onto encoding fixations
Match recall fixations to encoding using elastic matching algorithm

This repository contains the source code and dataset of the artilce "A consensus-based elastic matching algorithm for mappingrecall fixations onto encoding fixations in the looking-at-nothingparadigm". More information can be found on the [Project page](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/em.html).

## Setup 
- Install dependencies using pip/anaconda/etc, if needed:, ...
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
