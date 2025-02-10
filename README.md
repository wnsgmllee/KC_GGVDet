# Analysis of Key Conditions for Generalizable Generated Video Detection <br> 
###### 🏆 Received Honorable Mention at [KSC 2024] for our research [[paper]](https://github.com/myusername/myrepo/blob/main/paper.pdf)


<br><br>

## Project Summary
With the rapid advancements in generative AI, high-quality synthetic videos that are indistinguishable from real ones are increasingly prevalent online. When large generative models, which train on vast quantities of web data, unintentionally learn from synthetic data, it can distort the true data distribution and lead to *model collapse*. Therefore, it is critical to distinguish between real and synthetic data beforehand. Previous generative video detection models have struggled to generalize well to source models they were not specifically trained on. This project addresses this gap by identifying distinct features that differentiate synthetic videos from real ones, as verified through extensive experiments. Moreover, it presents a direction for leveraging these findings to improve the generalizability of detection models.

### Contribution
- Prior studies have suggested that traces of generative artifacts left on individual frames of synthesized videos can serve as cues for identifying them in model training; this study verifies the validity of this claim through the DIRE technique.
- We identify and present common characteristics found in generated video data used for model training.
- Using formulas and illustrations, we provide a rationale that these characteristics could play a significant role in improving generalization performance.
- Lastly, we discuss the relationship between the frames per second (FPS) of synthetic videos used in previous studies and their impact on generalization performance, proposing specific properties of video data that could enhance future research.

## Dataset
- **Image**: GenImage
  - Full Version: [GenImage Repository](https://github.com/GenImage-Dataset/GenImage)
  - Version used in this repo: [Tiny GenImage on Kaggle](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage)

- **Video**:
  - **Fake**: [DeMamba](https://github.com/chenhaoxing/DeMamba)
  - **Real**: [MSRVTT Dataset](https://arxiv.org/abs/2007.09049)

## Model
  - DIRE model ([DIRE Repository](https://github.com/ZhendongWang6/DIRE.git))  
    - **Acknowledgment**:  
      This repository utilizes the DIRE model as implemented by [Zhendong Wang et al.](https://github.com/ZhendongWang6/DIRE.git). 

## Files
- **compute_dire**: Generates DIRE frames by utilizing DDIM inversion on input data.
- **calc_DIRE**: Calculates the average pixel values of generated DIRE frames for each model.
- **to_freq**: Converts DIRE frames to the frequency domain using FFT.

