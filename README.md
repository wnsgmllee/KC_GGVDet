# 프로젝트 이름

## 프로젝트 개요
이 프로젝트는 다양한 이미지와 비디오 데이터셋을 이용하여 DIRE 모델과 관련된 연구 및 구현을 수행하는 것을 목표로 합니다.

## Datasets
- **Image**: Tiny GenImage 
  - Full Version: [GenImage Repository](https://github.com/GenImage-Dataset/GenImage)
  - 사용된 버전: [Tiny GenImage on Kaggle](https://www.kaggle.com/datasets/yangsangtai/tiny-genimage)
  
- **Video**: 
  - [DeMamba](https://github.com/chenhaoxing/DeMamba)
  - [MSRVTT Dataset](https://arxiv.org/abs/2007.09049)

- **Model**: 
  - DIRE 모델 ([DIRE Repository](https://github.com/ZhendongWang6/DIRE.git) 인용)

**사용된 생성 모델**: biggan, crafter, adm, glide, sdv5, lavie, real, sora

## Files
- **compute_dire**: DDIM inversion을 사용하여 DIRE 프레임 생성
- **calc_DIRE**: 각 모델에서 생성된 DIRE 프레임의 평균 픽셀 값을 계산
- **to_freq**: FFT를 사용하여 DIRE 프레임을 주파수 도메인으로 변환

## Repo 구축
1. 내 환경과 Git을 연결합니다.
2. [DIRE Repository](https://github.com/ZhendongWang6/DIRE.git)를 인용하는 방법을 확인하고 적용합니다.
3. 논문 내용을 추가하며, 이를 위해 GPT-4의 도움을 받습니다.
4. 프로젝트가 승인된 경우 해당 링크를 여기에 업데이트할 예정입니다.

## 설치 방법
1. 이 repository를 클론합니다.
   ```bash
   git clone https://github.com/username/my-repo.git
   cd my-repo
