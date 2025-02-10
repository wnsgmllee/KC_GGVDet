# Analysis of Key Conditions for Generalizable Generated Video Detection <br> 
###### 🏆 Honorable Mention, KSC 2024 
[[paper]](https://github.com/wnsgmllee/KC_GGVDet/blob/master/%EC%83%9D%EC%84%B1%EB%90%9C%20%EB%B9%84%EB%94%94%EC%98%A4%20%ED%83%90%EC%A7%80%EC%9D%98%20%EC%9D%BC%EB%B0%98%ED%99%94%20%EC%84%B1%EB%8A%A5%20%ED%96%A5%EC%83%81%EC%9D%84%20%EC%9C%84%ED%95%9C%20%ED%95%B5%EC%8B%AC%20%EC%A1%B0%EA%B1%B4%20%EB%B6%84%EC%84%9D%20(KSC%202024%20%EC%B5%9C%EC%A2%85%EB%B3%B8).pdf)


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

