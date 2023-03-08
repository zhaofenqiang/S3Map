# S3Map
This is the code for fast spherical mapping of cortical surfaces using [S3Map algorithm](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_16).
![Figure framework](https://github.com/zhaofenqiang/S3Map/blob/main/examples/fig_framework.png)

# Usage
1. Download or clone this repository into a local folder
2. Open a terminal and run the follwing code (better do this in a conda environment):
```
pip install  pyvista  tensorboard torch torchvision torchaudio
```
if only cpu is available, you can install the cpu version torch
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
3. Prepare your data, i.e., inner surfaces in vtk format (the file name should end in '.vtk')
4. Simply run "python S3Map_test.py -h" for the spherical mapping, and the expected output should be:
```
usage: S3Map_test.py [-h] --file FILE --hemi HEMI --folder FOLDER
                     [--config CONFIG] [--model_path MODEL_PATH]
                     [--device {GPU,CPU}]

S3Map algorithm for mapping a cortical surface to sphere

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  the full path of the inner surface in vtk format,
                        containing vertices and faces (default: None)
  --hemi HEMI, -hemi HEMI
                        the hemisphere of the input inner cortical surface
                        (default: None)
  --folder FOLDER, -folder FOLDER
                        a subject-specific folder for storing the output
                        results (default: None)
  --config CONFIG, -c CONFIG
                        Specify the config file for spherical mapping. An
                        example can be found in the same folder named as
                        S3Map_Config_3level.yamlIf not given, default is 3
                        level with 10,242, 40,962, 163,842 vertices,
                        respectively. (default: None)
  --model_path MODEL_PATH, -model_path MODEL_PATH
                        full path for finding all trained models (default:
                        None)
  --device {GPU,CPU}    The device for running the model. (default: GPU)
```
5. Use [paraview](https://www.paraview.org/) to visualize all generated .vtk surfaces, or [read_vtk](https://github.com/zhaofenqiang/S3Map/blob/a96c103f66db443ba52cdafee28af798a527fc54/sphericalunet/utils/vtk.py#L26) into python environment for further processing.

## Train a new model on a new dataset
After data prepration, modify the train.py file to match the training data in your own path. Then, run:
```
python S3Map_train.py
```

