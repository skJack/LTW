# Domain General Face Forgery Detection by Learning to Weight
This code belongs to the "Domain General Face Forgery Detection by Learning to Weight" 
## Installation
### Environment
Before run the code, you should run 
```
$ pip install -r requirements.txt
```
first.
### Dataset
This code is mainly focus on the GCD benchamrks, which contain three deepfake dataset:
Faceforensics++,CELEB-DF,DFDC (version for competition).
As for the Faceforensics++, you should extract face using MTCNN and prepare the folder as follows:
```
├── faceforensics++
  └──manipulated_sequences
    ├──Deepfakes
      └──c23
        └──mtcnn
          ├──000_003
            ├──000_003_0000.png
            ├──000_003_0001.png
            ├──...
          ├──001_870
          └── ...
      └──c40
        └──mtcnn
          ├──000_003
            ├──000_003_0000.png
            ├──000_003_0001.png
            ├──...
          ├──001_870
          └── ...
    ├──Face2Face
    ├──FaceSwap
    ├──NeuralTextures

  └──original_sequences
    └──youtube
      └──c23
        └──mtcnn
          ├──000
            ├──000_0000.png
            ├──000_0001,png
            ├──...
          ├──001
          └── ...
      └──c40
        └──mtcnn
          ├──000
            ├──000_0000.png
            ├──000_0001,png
            ├──...
          ├──001
          └── ...


```
And change the ffpp_original_path and ffpp_fake_path in config.py as your own data_root.
## Train
All the config is written in config.py. Before training, you can change the hyperparameters as you need.
Then run 
```
CUDA_VISIBLE_DEVICES = DEV_ID python train.py
```
to train the LTW model.
The log and model are saved in folder "result" automatically.
## Test
The config of the test is config_test.py. Before training, you should assign the path of tested model to the variable "model_path" in config_test.py.
Then you can run 
```
CUDA_VISIBLE_DEVICES = DEV_ID python test.py
```
to evaluate your model on all testset.