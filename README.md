# benchmarkDL

This repo contains script for a simple GAN model to benchmark its performance accross different platforms.    

### Benchmark
(for clarity this benchmarks the training duration only)

| Model                         | System   | Processor   | GPU          | Time Taken for the model to train | inference time |
|-------------------------------|----------|-------------|--------------|-----------------------------------|----------------|
| GAN-hands-100kbatch-100epochs | Personal | core i7     | Gtx 1060-6GB | 4.455 minutes                     |                |
| GAN-hands-100kbatch-100epochs | AWS      | gdn4.xlarge | T4-16GB      | 3.952 minutes                     |                |
| GAN-hands-100kbatch-100epochs | Personal | core i7     | -            | 25.348 minutes                    |                |

### Walk through
- [notebook/](https://github.com/ASH1998/benchmarkDL/tree/main/notebooks) - stores notebook format  which can be run directly.
- data/ - (to be created) this will store the downloaded data required for training.
- [src/](https://github.com/ASH1998/benchmarkDL/tree/main/src) - source code directory.
- [static/](https://github.com/ASH1998/benchmarkDL/tree/main/static) - contains image files.

### Requirements
- Download the data from [googleStorage API.](https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/hand.npy)
- python 3.7+
- tqdm (`pip install tqdm`)
- numpy (`pip install numpy`)
- sklearn (`pip install -U scikit-learn`)
- matplotlib (`pip install matplotlib`)
- tensorflow (`pip install tensorflow
`)

### Execution using Script
(after all the requirements are installed)
1. Download or clone this repo.
2. `mkdir data`
3. `wget https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/hand.npy` (not needed if data has already been downloaded)
4. store the `hand.npy` dataset in the `data/` directory.
5. `cd src`
6. `python model.py`

### Execution using Notebook
- notebook with full execution has been provided in [this directory](notebooks).            
- Any of the two notebook can be run independently after installing required packages.

### Results 
1. Personal PC: 
![1060](https://github.com/ASH1998/benchmarkDL/blob/main/static/zoomin-1060gtx-predator.jpg)

- 
2. AWS
![AWS](https://github.com/ASH1998/benchmarkDL/blob/main/static/zoomin-aws-gdn4xlarge.jpg)
### Disclaimer
This repo is just for quick benchmarking.
Running the above model for only 100 epochs wont yeild good results.            
For good results epoch should be around 500, and batchsize ~ 1000
