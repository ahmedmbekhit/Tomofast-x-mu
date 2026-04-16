<img width="2096" height="1178" alt="worflow_git" src="https://github.com/user-attachments/assets/a78433a2-6933-4170-afc6-97106d96a18a" />

# Tomofast-x-µ: 3-D Magnetization Reconstruction of Micro-Magnetic Imaging
The repositry includes the codes and data of (Bekhit, A. M., Ogarko, V., Kirscher, U., Liu, Y., Jessell, M., Li, Z.-X., & Evans, K. (2026). Tomofast-x-µ: 3-D magnetization reconstruction of micromagnetic imaging (Version 1) [Preprint]. https://doi.org/10.22541/essoar.15001802/v1 )


## Description
This  is an open-source Python code based on the [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x) inversion code (Giraud et al., 2021; Ograko et al., 2024). It is subdivided into three major and independent tools:
1.	Three-dimensional micromagnetic modelling: perform full 3-D magnetic vector inversion to recover a full 3D image of the QDM scan.
2.	Magnetic source locations detection: run a series of fast micro-scale inversions within the grain-discretized 3D volume to detect the magnetic source's true locations.
3.	Paleomagnetic information delineation: fit the micro-mag data at the locations of detected magnetic sources to recover magnetization vectors, then forward-calculate magnetization direction (declination and inclination) and amplitude (intensity).
   
## Installation
To use this code you must install Tomofast-x inversion code: [Tomofast-x Install](https://github.com/TOMOFAST/Tomofast-manual/blob/main/install.md)  
For Tomofast-x detailed user manual [Tomofast_Manual](https://github.com/TOMOFAST/Tomofast-manual) 

## Dependencies
Data handling and numerical calculations were carried out using NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), and pandas (McKinney, 2011). Grid-based data and input files were managed with xarray (Hoyer and Hamman 2017) and pathlib. Visualization and figure generation were performed using matplotlib (Hunter, 2007), and Paraview for 3D (Ahrens et al., 2005). Paleomagnetic data analysis used PmagPy (ipmag) (Tauxe et al., 2016). File handling, pattern matching, and system operations used the standard Python libraries os, sys, glob, subprocess, re, and math. Automatic anomaly detection and 3-D grain extraction relied on functions from scipy.ndimage, including labeling, dilation, and structure generation.

## Simple instructions
for using this code we recommend reproducing the most recent version (Tomofast-x-µ.ipynb) above , with your own data and updated parameters, it has detailed instructions. 
you need firstly to install Tomofast-x code and then copy the code, and run it with Bz.Mat/Bz.Csv file. 
This code also could be used as python interface to help Tomofast-x usage.


## Authors and contacts
Ahmed Bekhit, Vitaliy Ogarko, Mark Jessel, Yebo Liu, Katy Evans, Zheng-Xiang Li, Uwe Kirscher

all questions are welcome Ahmed Bekhit: a.hussein4@postgrad.curtin.edu.au or ahmed.m.bekhit@alexu.edu.eg 
