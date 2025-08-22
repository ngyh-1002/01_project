# 🚌 무인 자율주행버스 휠체어 인식 시스템 개발 프로젝트

## 1\. 프로젝트 개요

본 프로젝트는 무인 자율주행 버스에 휠체어 탑승객을 위한 인식 및 편의 기능을 추가하는 것을 목표로 합니다. [cite\_start]휠체어가 감지되면 관리자 개입 없이 자동으로 경사로(램프)를 작동시켜 휠체어 이용객의 버스 탑승 편의성과 자율주행 버스의 접근성을 향상시키고자 합니다[cite: 7, 8].

## 관련제품 (링크)
- [버스 경사로](https://kr.made-in-china.com/co_czxinder/product_Disabled-Electric-Aluminum-Wheelchair-Ramp-for-Low-Floor-Bus-with-350kg-Loading-EWR-_eghirugng.html)
# 하드웨어
- 웹캠
- 데스크 탑 
# 진행상황
- [0818 피드백진행](/feedback/0818.md)
- [0819 피드백진행](/feedback/0819.md)
- [0820 피드백진행](/feedback/0820.md)
- [0821 피드백진행](/feedback/0821.md)
- [0821_1 피드백진행](/feedback/0821_1.md)

## 3\. 기술 구현 및 정확도 분석

### 3.1. 휠체어 탐지 구현 (Roboflow API)

  * [cite\_start]Roboflow의 API(`https://universe.roboflow.com/mobilityaids/wheelchair-detection-hh3io/model/3`)를 활용하여 휠체어 탐지 기능을 구현했습니다[cite: 60, 61].

### 3.2. 문제점 및 개선 방안

  * [cite\_start]**문제점**: 초기 모델의 탐지율은 약 85.7%로 실제 서비스에 적용하기에는 부적합했습니다[cite: 83, 85, 86]. [cite\_start]이는 모델 학습에 사용된 데이터셋의 한계 때문으로 판단되었습니다[cite: 87, 88].
  * [cite\_start]**개선 방안**: 새로운 데이터셋을 사용해 모델을 재학습하고, 신뢰도 임계값(Confidence Threshold)을 조정하여 탐지율을 최적화하는 방안을 수립했습니다[cite: 94, 95].

### 3.3. 모델 재학습 및 교차 검증

  * [cite\_start]새롭게 학습된 모델(`https://app.roboflow.com/ngyh1002/wheelchair-detection-hh3io-gpvvj/models/wheelchair-detection-hh3io-gpvvj/1`)을 적용하여 테스트를 진행했습니다[cite: 100, 116, 117, 118, 121, 122].
  * [cite\_start]하지만 오히려 정확도가 약 74.6%로 떨어지는 문제가 발생했습니다[cite: 156, 158].
  * [cite\_start]이 문제를 해결하기 위해 기존 모델과 재학습 모델을 동일한 데이터셋에 적용하여 교차 검증을 실시했습니다[cite: 161, 169].
  * [cite\_start]교차 검증 결과, 두 모델은 특정 조건에서 서로 다른 성능을 보였으며, 단순히 하나의 모델을 선택하기보다 두 모델을 함께 활용하는 방안을 고려하게 되었습니다[cite: 192, 193].

## 4\. 최종 시스템 로직 및 해결책

### 4.1. API 병목 현상 문제

  * [cite\_start]두 모델을 영상으로 교차 검증하는 과정에서 API 병목 현상이 발생하여 실시간 영상 처리 속도가 현저히 느려지는 문제가 발생했습니다[cite: 200, 201]. [cite\_start]이는 실시간성이 중요한 자율주행 시스템에 부적합하다는 결론으로 이어졌습니다[cite: 202].

### 4.2. 해결책: 로컬 모델 도입

  * [cite\_start]외부 API 통신 없이 로컬에서 직접 구동 가능한 휠체어 탐지 모델인 `best.pt`를 도입했습니다[cite: 204, 206, 230].
  * [cite\_start]이 모델의 정확도 테스트 결과, 약 90.7%의 탐지율을 보였습니다[cite: 242].
  * [cite\_start]이를 통해 API 지연 문제를 해결하고, 더 빠르고 안정적인 실시간 휠체어 탐지가 가능할 것으로 기대됩니다[cite: 244].

### 4.3. 최종 시스템 로직

  * [cite\_start]**버스 대기 로이 (ROI, Region of Interest) 설정**: 버스가 정차하는 특정 구역을 '버스 대기 로이'로 설정합니다[cite: 254].
  * [cite\_start]**휠체어 탐지 및 교차 비율 계산**: '휠체어 로이'가 '버스 대기 로이'와 겹치는 비율을 계산합니다[cite: 256].
  * [cite\_start]**슬로프 자동 작동**: 교차 비율이 30%를 초과하면 휠체어가 탑승하려는 것으로 판단, 슬로프를 자동으로 하강시키는 신호를 보냅니다[cite: 258]. [cite\_start]이 로직은 불필요한 슬로프 작동을 방지하고 시스템 효율성을 높입니다[cite: 259].


## 5\. Python 모듈 설치

[cite\_start]프로젝트에 필요한 모든 모듈은 아래 명령어를 통해 설치할 수 있습니다[cite: 261].

```bash
pip install inference-sdk opencv-contrib-python ultralytics matplotlib Pillow
```

  * [cite\_start]`inference-sdk`: Roboflow Inference Server와 통신하여 객체 탐지 추론을 수행합니다[cite: 262].
  * [cite\_start]`opencv-contrib-python`: OpenCV 라이브러리 전체 버전입니다[cite: 263].
  * [cite\_start]`ultralytics`: YOLOv8 모델을 로컬에서 실행하고 추론하는 데 필요합니다[cite: 263].
  * [cite\_start]`matplotlib`: 그래프 및 시각화 자료를 만드는 데 사용됩니다[cite: 264].
  * [cite\_start]`Pillow`: 이미지 처리용 라이브러리입니다[cite: 265].

## 6\. 개발 도구

  * ChatGPT (4)
