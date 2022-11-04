---
title: "논문 공부: Deep Neural Networks for YouTube Recommendations(2016)"
excerpt : 유튜브 추천 딥러닝 논문 공부
categories:
  - Click-Through Rate Prediction
  - 논문 리뷰
toc: true

---

[논문](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)을 읽고 공부하는 글이다.

---

# 1. _INTRODUCTION_
---

|![그림 1](\assets\images\다양한 공부\논문\CTR\Deep Neural Networks for YouTube\그림 1.png)|
|:-:|
|그림 1|

그림 1은 유튜브 앱에서 추천을 하는 모습을 보여준다. 

유튜브 영상을 추천하는 것은 3개의 주요 관점에서 매우 어렵다. 
* Scale 

  이미 존재하는 추천 알고리즘들은 유튜브을 위해선 작동하지 않는다. 따라서 유튜브의 방대한 유저 기반 말뭉치(corpus)를 다루기 위한 고도로 특화된 distributed learning 알고리즘과 효율적인 serving system들이 필요하다.


* Freshness :

  유튜브는 매 초마다 업로드되는 몇시간 단위의 영상에서 다이나믹한 말뭉치들을 갖는다. 그래서 추천 시스템은 새롭게 업로드된 내용과 유저에 의해 취해지는 최신 action들을 
  충분히 모델링 할 수 있을 정도로 반응해야한다. 

* Noise : 

  유튜브에서 historical user behavior은 예측하기 어렵다. 왜냐하면 sparsity와 관측할 수 없는 외부적인 요인들의 다양성때문이다. 그리고 사용자 만족의 ground truth를 얻지 못하고 대신 
  노이즈가 낀 feedback signal들을 모델링한다.

그래서 유튜브는 대부분의 학습 문제들을 해결하기 위해 딥러닝 쪽으로 방향을 틀고 있다. 

본 논문은 간단한 시스템 개요에 대해 먼저 소개하고(Section 2), Section 3에서 candidate generation model에 대해 설명한다. 학습 방법과 추천 방법에 대해서도 같이 설명한다. 
Section 4는 ranking model을 상세히 설명하고 고전적인 로지스틱 회귀가 expected watch time을 예측하는 모델을 학습하기 위해 어떻게 수정되는지 보여줄 것이다. 그리고 실험 결과가 이런 상황에서
은닉 층의 깊음이 얼마나 도움이 되는지도 보여줄 것이다.
실혐 결과는 은닉 층의 깊음과 추가적인 여러 종류의 시그널들로 부터 어떻게 이득을 취하는지 보여줄 것이다.

<br/><br/>

# 2. _SYSTEM OVERVIEW_
---

|![그림 2](\assets\images\다양한 공부\논문\CTR\Deep Neural Networks for YouTube\그림 2.png)|
|:-:|
|그림 2|

위 그림은 본 추천 시스템의 전체적인 구조를 보여준다. 이 시스템은 **candidate generation**과 **ranking**을 위한 2개의 뉴럴 넷으로 구성되어 있다.(그림의 파란색 부분)

* **candidate generation network**

  **candidate generation 네트워크**는 입력으로 사용자들의 유튜브 활동 history를 사용하고(수 백만개), 이 엄청난 corpus에서 비디오의 작은 subset(수 백개)을 추출한다.
  그리고 collaborative filtering을 통해서 광범위한 개인화?(personalization)를 제공한다. {이게 무슨 말인지 모르겠다.} 사용자 간의 유사성은 비디오 시청의 ID들, 검색 쿼리 토큰, 통계와 같은 coarse한 feature로 표현된다. 

* **ranking network**

  몇 몇의 최고의 추천을 제공하려면 high recall{아마 평가 지표를 얘기하는 것 같다}을 지닌 후보들 사이에 상대적 중요성을 구분하기 위한 fine-level representation이 필요하다. 
  **ranking network**는 이 일을 각각의 비디오들에 score를 할당함으로써 수행한다. 그래서 높은 score를 지닌 것이 추천되는 시스템인 것 같다.

그리고 이런 방식의 네트워크 설계는 다른 소스로부터 생성된 candiate들을 섞을 수 있게 해준다. 이런 방식은 [3]에서 이미 언급됬다.

음.. 학습을 위해선 여러 평가 지표들(precision,recall,ranking loss...)들을 사용하지만, 
알고리즘의 효과를 최종적으로 결정하기 위해선 live experiments를 통한 A/B 테스팅을 확인한다. live experiment에서 우린 클릭 률, 시청 시간 등의 미묘한 변화를 측정할 수 있다.


<br/><br/>

# 3. _CANDIDATE GENERATION(후보 생성)_
---

후보 생성 네트워크에서 방대한 유튜브 corpus는 사용자와 관련된 수백개의 비디오들로 작아진다.  

기존의 추천 시스템들은 rank loss에 따라 훈련된 matrix factorization 방법이었다. 우리의 뉴럴 넷도 factorization 기술의 비선형적 일반화라고 볼 수 있다.

## 3.1. Recommendation as Classification

사용자 U와 컨텐츠 C에 기반한 코퍼스 V로부터 