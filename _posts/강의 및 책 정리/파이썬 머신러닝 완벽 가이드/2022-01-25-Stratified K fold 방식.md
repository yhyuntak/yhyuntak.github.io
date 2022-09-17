---
title: "train set, validation set, test set"
categories:
  - 파이썬 머신러닝 완벽 가이드
toc: true
---

학습을 위해선 데이터 셋을 train set과 test set으로 나눠서 사용하는 것이 일반적입니다. 보통 비율을 7:3 혹은 8:2로 두죠. 

이 때, 사이킷 런의 model_selection 에서 train_test_split을 사용합니다. 

그러나 나누어진 set들로만 학습을 한다면, 데이터 셋들이 모든 데이터 셋을 대표한다고 볼 수 있을까요? 전혀 새로운 데이터가 들어올 경우, 제대로된 예측을 못할지도 모릅니다.
즉 훈련 데이터 셋 하나만 가지고 테스트 셋을 목표로 학습을 한다면 과적합(overfitting)이 일어날 수 있습니다. 그래서 우리는 **validation set**을 만들어 train set으로 학습하고, validation set으로
검증을 한 후, 모델이 학습이 다 되면 test set으로 결과를 확인해볼 수 있는 것이죠. 검증 데이터 셋을 만드는 방법 중 가장 대표적인 것이 **K fold Cross Validation** 입니다.

# K fold Cross Validation

K fold Cross Validation은 간단합니다. train set을 K개로 나눠서 K번 학습을 해보는 겁니다. 

첫번째 데이터 셋을 validation set으로 두고 나머지 K-1개의 데이터 셋을 train set으로 둔 후, 학습을 진행합니다. 
두번째, 세번째, ... K번째까지 순서대로 validation set으로 두고 총 K번의 학습을 진행하고, 오차율을 평균내어 결과를 확인하면 됩니다.

하나 주의해야 할 점은 K fold Cross Validation으로 데이터들을 나눌 때, label의 분포를 신경쓰지 않기 때문에 잘못된 학습을 할 수 있다는 것입니다. 
2개의 클래스를 갖는 데이터 셋을 예로 들면, K개로 쪼개었더니 어느 한쪽에만 1개의 클래스가 다 몰려있는 겁니다.. 그렇다면 학습이 잘 될 수 없겠죠.
따라서 데이터 내의 label의 분포도를 파악하여 적절하게 fold 하는 것이 중요한데, 이 때 사용하는 방식이 Stratified K fold 방식입니다. 
