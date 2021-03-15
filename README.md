# SKFont

## Introduction

This is the Tensorflow implementation of the **SKFont: Skeleton-Driven Korean Font Synthesis with Conditional Deep Adversarial Networks**.

[paper will be available soon](https://github.com/ammar-deep/SKFont/blob/main/README.md)

## Abstract
Will be added soon

## Model Architecture
![Architecture](imgs/architecture.png)

<!-- ## Some Results

<!--### SKFont results on Gothic and Ming font styles
![comparison](imgs/SKFont_results_more.png)

<!--### SKFont results on unseen cross languages
![cross_languae](imgs/cross_language.png)

<!--### SKFont results on cursive font styles
![cross_languae](imgs/cursive.png) -->

## Prerequisites

- Windows
- CPU or NVIDIA GPU + CUDA cuDNN
- python 3.6.8
- tensorflow-gpu 1.13.1
- pillow 6.1.0 

## Get Started

### Installation

#### Setting up the environment
1. ```
   conda create --name tutorial-TF python=3.6.8
   ```
2. ```
   conda activate tutorial-TF or activate tutorial-TF
   ```
3. ```
   conda install -c anaconda tensorflow-gpu=1.13.1
   ```
4. ```
   conda env update --file tools.yml
   ```

### Datasets
Our model consists of three sub models namely F2F-F2S-S2F. For each model we have to prepare a paired dataset. i.e. a source to target font paired dataset, a target font to corresponing skeleton dataset, and a target skeleton to corresponding font dataset. 
To do this place any korean font in scr_font directory and N number of target fonts in the trg_font directory. Then run the below commands for data preprocessing.

1. Generate Source font images
    ```
    python ./tools/src-font-image-generator.py
    ```
    
2. Generate Target font images
    ```
    python ./tools/trg-font-image-generator.py
    ```
    
3. Generate Target font skeleton images
    ```
    python ./tools/trg-skeleton-image-generator.py
    ```
    
4. Combine source, target, and target skeletons
    ```
    python ./tools/combine_images.py --input_dir src-image-data/images --b_dir trg-image-data/images --c_dir skel-image-data/images --operation combine
    ```
    
5. Convert images to TFRecords
    ```
    python ./tools/images-to-tfrecords.py
    ```
    
 ### Training the model
 
 #### Pre-training the model
 ```
 python main.py --mode train --output_dir trained_model --max_epochs 25 
 ```
 
 #### Finetuning the model
 To learn an unseen font style you can fine tune an already pre-trained model with the below command. If you want to generate the already learnt font styles just skip the below command.
 
 ```
 python main.py --mode train --output_dir finetuned_model --max_epochs 500 --checkpoint trained_model/ 
 ```
 
 ### Testing the model
 
Generate images just like before but this time use a different module for creating testing TFRecords with the below mentioned command.

1.  Convert images to TFRecords
    ```
    python ./tools/test-images-to-tfrecords.py
    ```
#### Generating results
 ```
python main.py --mode test --output_dir testing_results --checkpoint finetuned_model
 ```

## Acknowledgements

This code is inspired by the [pix2pix tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) project.

Special thanks to the following works for sharing their code and dataset.

- [tensorflow-hangul-recognition](https://github.com/IBM/tensorflow-hangul-recognition)
- [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow)

## Citation

Citation will be added soon. Please cite our work if you like it. 

## Copyright

The code and other helping modules are only allowed for PERSONAL and ACADEMIC usage.
