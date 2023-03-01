# S3Map
This is the code for fast spherical mapping of cortical surfaces using [S3Map algorithm](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_16).

## Usage
1. Download or clone this repository into a local folder
2. open a terminal and run the follwing:
```
pip install  pyvista  tensorboard torch torchvision torchaudio
pip install --prefix=/proj/ganglilab/users/Fenqiang/sunetpkg git+https://github.com/Deep-MI/LaPy.git#egg=lapy
```
if only cpu is available, you can install the cpu version torch
```
pip install --prefix=/proj/ganglilab/users/Fenqiang/sunetpkg torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
3. Prepare your data, i.e., inner surfaces in vtk format (the file name should end in '.vtk')
4. Simply run "python S3Map_test.py -h" for the spherical mapping, and the expected output should be:
```
usage: S3Map_test.py [-h] --file FILE [--folder FOLDER] --config CONFIG
                     [--model_path MODEL_PATH] [--device {GPU,CPU}]

S3Map algorithm for mapping a cortical surface to sphere

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  the full path of the inner surface in vtk format,
                        containing vertices and faces (default: None)
  --folder FOLDER, -folder FOLDER
                        a subject-specific folder for storing the output
                        results (default: None)
  --config CONFIG, -c CONFIG
                        Specify the config file for spherical mapping. An
                        example can be found in the same folder named as
                        S3Map_Config_3level.yaml (default: None)
  --model_path MODEL_PATH
                        full path for finding all trained models (default:
                        None)
  --device {GPU,CPU}    The device for running the model. (default: GPU)
```
