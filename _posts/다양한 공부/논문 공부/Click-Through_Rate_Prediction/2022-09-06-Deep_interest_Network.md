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

그래서 저자들은 이 모델 Deep Interest Network(DIN)을 제시해 문제를 타파해보려고 합니다. 이 모델은 특정한 광고에 대해 historical behaviors로 부터 유저 관심도의 표현을 배우기 위해 local activation unit을 
설계합니다. 

<br/>

---

# 1. _Introduction_

* FCN은 end-to-end 학습을 진행하고, pixels-to-pixels 학습을 진행함
* 저자들의 방법은 전,후처리 따위 없이 진행될 것이고 fully convolutional and fine-tuning 을 통해 
좋은 분류기로써의 성능을 보여줌
* Semantic segmentation에서, global information은 이 물체가 **무엇**인지 판단하는데 도움을 주고, local information은 이 물체가 **어디**에 있는지 판단하는데 도움을 줌
* Deep feature hierachies는 location과 semantic을 local-to-global pyramid구조에서 같이 해석하는데, 저자들은 이런 정보들을 합치고 싶어함.
* 그래서 일명 **skip** 구조를 정의해서 deep,coarse,semantic 정보들과 shallow,fine,appearance 정보들을 합치고자 한다.

<br/>

---
