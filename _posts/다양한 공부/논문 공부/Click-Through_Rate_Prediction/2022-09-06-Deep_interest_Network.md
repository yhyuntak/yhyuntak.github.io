---
title: "논문 공부: Deep Interest Network for Click-Through Rate Prediction(2018)"
excerpt : DIN 논문 공부
categories:
  - Click-Through Rate Prediction
  - 논문 리뷰
toc: true

---

[본 논문 FCN](https://arxiv.org/abs/1706.06978?context=cs)을 읽고 공부하는 글입니다.

---

해당 분야에 첫 발을 들여놓는 만큼 abstract부터 꼼꼼히 읽고자 합니다.

# 0. _Abstract_

이쪽 분야의 딥러닝 기술들은 similar Embedding과 MLP를 사용하는 듯 합니다. 이 방법들은 크기가 큰 scale sparse input feature들을 낮은 차원의 embedding vector들로 매핑하고, 
group-wise 방법으로 고정된 길이의 백터들로 변환됩니다. 마지막으로 이 두개를 결합해 multi-layer perceptron(MLP)으로 보내 feature들 간의 비선형적 관계를 학습합니다. 
이 과정 속에서, 사용자의 특징들은 후보 광고군이 무엇이든 상관없이 고정된 길이의 representation vector로 압축됩니다. 사실 이 부분이 bottleneck이 되어 많은 historical behavior들로부터  
사용자들의 다양한 어떤 흥미를 효과적으로 팍! 캡쳐하기 위한 Embedding&MLP 방법에 문제를 일으키게 됩니다. 

그래서 저자들은 이 모델 Deep Interest Network(DIN)을 제시해 문제를 타파해보려고 합니다. 이 모델은 특정한 광고에 대해 historical behaviors로 부터 유저 관심도의 representation vector를 배우기 위해 local activation unit을 
설계합니다. 이 vector는 다른 광고들에 따라 변화하고, 모델의 표현 능력을 엄청나게 향상시킵니다. 

게다가 저자들은 2개의 기술을 개발합니다.

* mini-batch aware regularization
* data adaptive activation function

  이것들은 아직은 잘 모르겠지만, 학습할 때 파라미터의 개수에 관련해 영향을 주는 것 같습니다.

데이터 셋으로는 2개의 공공 데이터와 Alibaba real production dataset들을 사용해 저자들의 모델의 효과를 입증하고자 합니다.
실제로 논문이 쓰여질 시점엔 DIN은 이미 알리바바의 online display advertising system에 성공적으로 사용했다고 합니다.

<br/>

---

# 1. _Introduction_




<br/>

---
