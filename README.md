# NPI-HGNN

NPI-HGNN is proposed based on the link prediction idea of "extracting closed subgraphs + graph classification", and is a heterogeneous information network embedding method for accurately predicting potential NPIs. NPI-HGNN mainly includes five parts: negative sample generation, closed subgraph extraction, node feature representation, graph representa-tion, and NPI prediction. 
    
    Note: The dependent library and version information are in requirements.txt

## 1. Running process

### 1.1 Generating kmer ncRNA-ncRNA similarity network

>Python .\src\npi_hgnn\generate_rrsn.py --dataset {datasetName} --ratio {ratio}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --ratio : The edge that generates the ncRNA-ncRNA similarity network accounts for the proportion of min(NPIs,PPIs). [0-1]

### 1.2 Generating kmer protein-protein similarity matrix

>Python .\src\npi_hgnn\generate_ppsm.py --dataset {datasetName} --ratio {ratio}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]

### 1.3 Negative sample selection and partitioning the dataset

>Python .\src\npi_hgnn\generate_edgelist.py --dataset {datasetName} --samplingType {no} --num_fold {num}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --samplingType : [0 | 1 | 2 | 3].  
* --num_fold : [integer greater than 0]

### 1.4 Extracting kmer substrings of ncRNA and protein sequences

>Python .\src\npi_hgnn\generate_kmer.py --dataset {datasetName} --kRna {num} --kProtein {num}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --kRna : [3].  
* --kProtein : [2]

### 1.5 Generating kmer occurrence vector

>Python .\src\npi_hgnn\generate_frequency.py --dataset {datasetName}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]

### 1.6 Generating pyfeat vector

>Python .\src\npi_hgnn\generate_pyfeat.py --dataset {datasetName}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --other_parameters : please see <https://github.com/mrzResearchArena/PyFeat>

### 1.7 Generating node feature vector

>Python .\src\npi_hgnn\generate_node_vec.py --dataset {datasetName} --nodeVecType {no} --samplingType {no} --subgraph_type {no} --fold {no}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --nodeVecType : node feature type. [0 | 1 | 2].  
* --samplingType : [0 | 1 | 2 | 3].  
* --subgraph_type : [0 | 1 | 2].  
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  
* --other_parameters : please see <https://github.com/aditya-grover/node2vec>

### 1.8 Generating dataset

>Python .\src\npi_hgnn\generate_dataset.py --dataset {datasetName} --nodeVecType {no} --samplingType {no} --subgraph_type {no} --fold {no}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --nodeVecType : node feature type. [0 | 1 | 2].  
* --samplingType : [0 | 1 | 2 | 3].  
* --subgraph_type : [0 | 1 | 2].  
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  

### 1.9 Training

>Python .\src\npi_hgnn\train.py --dataset {datasetName} --nodeVecType {no} --samplingType {no} --subgraph_type {no} --fold {no}

#### Parameters

* --dataset : Dataset name. [NPHN3265 | NPHN4158 | NPHN7317 | NPHN-Homo | NPHN-Mus]
* --nodeVecType : node feature type. [0 | 1 | 2].  
* --samplingType : [0 | 1 | 2 | 3].  
* --subgraph_type : [0 | 1 | 2].  
* --fold : which fold is this. [0 | 1 | 2 | 3 | 4].  
* --other_parameters : Taking default value. 

## 2. Case study

### 2.1 Generating case study dataset

>Python .\src\case_study\case_study_dataset.py

#### Parameters

* --all_parameters : Taking default value. 

### 2.2 Case study training

>Python .\src\case_study\case_study_train.py

#### Parameters

* --all_parameters : Taking default value. 

### 2.3 Case study prediction

>Python .\src\case_study\case_study_predict.py --epoch {no}

#### Parameters

* --epoch : [integer greater than 0].  
* --other_parameters : Taking default value. 




