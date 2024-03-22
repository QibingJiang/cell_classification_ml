# cell_classification_ml
# Introduction
This repository is the implementation fo paper "Machine Learning Based Cell Classification for Bright
Field Time-Lapse Images".

This application contains three components: Cell Detection, Cell Tracking and Cell Classification.
The overall workflow of the framework is as follows:
![workflow](https://github.com/QibingJiang/cell_classification_ml/assets/63761274/cfb8c053-0bf1-4afb-942b-cdca2e66070e)

The result of the proposed method compared with KTH-SE\(Cell tracking Chellenge\) and YOLO V8 is as followed:

![Beacon_A](https://github.com/QibingJiang/cell_classification_ml/assets/63761274/3282ad40-496c-46cb-a4a8-7750ac17ef3e)

# Running
The work depends on 
```
pip install opencv_python
pip install scipy
pip install astropy
pip install shutil
pip install tensorflow
```
Run the detection, tracking and classification by the command:
```
python3 cell_process.py /input_dir/ /output_dir/
```
