---
title: "논문 공부: Deep Interest Network for Click-Through Rate Prediction(2018)"
excerpt : DIN 논문 공부
categories:
  - Click-Through Rate Prediction
  - 논문 리뷰
toc: true

---

[논문](https://arxiv.org/abs/1706.06978?context=cs)을 읽고 공부하는 글입니다.

---

해당 분야에 첫 발을 들여놓는 만큼 abstract부터 experiment까지 꼼꼼히 전부 해석하면서 읽고자 합니다.

# 0. _Abstract_
---

이쪽 분야의 딥러닝 기술들은 similar Embedding과 MLP를 사용하는 듯 합니다. 이 방법들은 크기가 큰 scale sparse input feature들을 낮은 차원의 embedding vector들로 매핑하고, 
group단위로 고정된 길이의 백터들로 변환됩니다. 마지막으로 이 두개를 결합해 multi-layer perceptron(MLP)으로 보내 feature들 간의 비선형적 관계를 학습합니다. 
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

<br/><br/><br/>


# 1. _Introduction_

---
cost-per-click(CPC) 광고 시스템에서, 광고들은 eCPM(effective cost per mille)에 의해 순위가 매겨집니다. 여기서 eCPM은 입찰 가격과 CTR(click-through rate)의 곱으로 표현됩니다. 
CTR은 시스템에 의해 예측될 필요가 있습니다. 이런 이유로, CTR 예측 모델의 성능은 최종 수입에 직접적으로 영향을 미치고, 광고 시스템에서 핵심 역할을 맡습니다. CTR 예측을 모델링하는 것은 연구계와 산업계 모두한테 큰 관심사입니다.

최근에, [컴퓨터 비젼](https://arxiv.org/abs/1608.06993)과 [NLP](https://arxiv.org/abs/1409.0473)에서의 딥러닝의 성공으로인해, 딥러닝 기반의 방법들이 CTR을 예측하는 방법에 대한 연구[3,4,21,26]가 발표되고 있습니다.
이 연구들은 similar Embedding&MLP 패러다임을 따릅니다. 이것들은 크기가 큰 scale sparse input feature들을 낮은 차원의 embedding vector들로 매핑하고, 
group단위로 고정된 길이의 백터들로 변환됩니다. 마지막으로 이 두개를 결합해 multi-layer perceptron(MLP)으로 보내 feature들 간의 비선형적 관계를 학습합니다. 
[로지스틱 회귀 모델](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/41159.pdf)과 비교해서, 위 방법들은 많은 feature engineering job들을 줄일 수 있고,
model capability를 크게 향상시킬 수 있습니다. 앞으로 이 논문에선 위 방법들은 간단히 Embedding&MLP라고 부르겠습니다.(이 리뷰 글에선 E&M이라고 하겠습니다.) 이것은 CTR 예측 task에서 요즘(당시에) 가장 인기가 많습니다. 

그러나, dimension이 제한된 user presentation vector는 E&M 방법들에서 사용자들의 다양한 interest들을 표현하는데 문제가 될 것입니다(bottleneck). 이커머스 사이트에서 광고를 전시하는 것을 한번 예제로 봅시다.
사용자들은 이커머스 사이트를 방문하고 있을 때, 다른 종류의 물건들을 **동시에** 관심있게 볼지도 모릅니다. **이것을 user interest들이 *다양하다*라고 이야기할 수 있습니다.** {이 개념은 확실히 알고 갑시다.} 
CTR 예측 task에 대해선, 사용자의 관심사들(user interests)은 보통 사용자 행동 데이터(user behavior data)로부터 포착됩니다. E&M 방법들은 특정한 유저들에 대한 모든 관심사들의 representation을 사용자 행동들의 embedding vector들을
고정된 길이의 벡터(fixed-length vector)로 변환하면서 학습합니다. 이 고정된 길이의 벡터는 모든 유저들의 representation vector들이 표현되는 유클리디안 공간에 있습니다. 
**다시 말해서, 사용자의 다양한 관심사들은 고정된 길이의 벡터로 압축된다는 것입니다.** 고정된 길이의 벡터는 E&M 방법들의 표현 능력을 제한합니다. 
그래서! 사용자들의 다양한 관심사들을 충분히 표현해주는 representation capable을 만들기 위해선, 고정된 길이의 벡터의 dimension은 크게 확장될 필요가 있습니다! {표현력을 제한하니까 차원을 키우자는 의미인듯}
그러나.. 불행히도 확장했다가는 학습 파라미터들의 크기도 커질거고 제한된 데이터하에 과적합의 위험성이 커지게 될겁니다. 게다가, 메모리와 계산량에 부담을 주니까 이건 실제 산업현장에서 온라인 시스템으로 사용 할 수 없습니다.

반면에, 후보 광고을 예측할 때, 특정 사용자의 모든 다양한 관심사들을 같은 벡터로 압축할 필요가 없습니다. 왜냐하면, 오직 사용자들의 관심사의 일부분만이 그들이 클릭할지 안할지에 영향을 미치기 때문입니다. 
예를 들면, 여성 수영 선수는 저번주에 신발보다 수영복을 샀기 때문에 추천되는 고글을 클릭할 것입니다. 이 것에서 영감을 받아, 저자들은 Deep Interest Network(DIN)이라는 네트워크를 제안합니다! DIN은 후보 광고가 주어진 
historical behavior들의 관련성을 고려하면서 사용자 관심사들의 representation vector를 적응적으로(adaptively) 계산합니다. {음.. 직역하면 이런데.. 아마 historical behavior라는 것은 이 사람이 무슨 행동을 해왔는지에 대한
것을 나타내는 것 같다. 이 사람이 해온 행동에 따라 후보 광고가 주어져야 이 사람이 클릭하지 않겠는가?} local activation unit(?)을 도입함으로써 DIN은 historical behaviors에 관련있는 부분들에 대해 soft-searching하면서 관련된 유저의 관심사들에 주목하고,
후보 광고에 대한 유저 관심사들의 representation을 얻기 위해 weighted sum pooling을 합니다. **후보 광고와 높은 관련성을 보이는 behavior들은 아주 활성화된 weight들을 얻고 유저 관심사들의 representation을 지배합니다.** 
저자들은 이런 현상을 실험 세션에서 시각화합니다. 이 방법으로 유저 관심사들의 representation vector는 다른 광고들에 따라 다릅니다. 이것은 제한된 차원에서 모델의 표현 능력을 향상시키고, DIN이 사용자들의 다양한 관심사들을 더 잘 포착하게 해줍니다!

large scale sparse feature들을 가진 산업의 deep network를 학습하는 것은 굉장히 어렵습니다. 예를 들면, SGD 기반의 최적화들은 각 미니 배치에서 나타나는 sparse feature들의 파라미터들만 업데이트합니다. 
그러나 전통적인 $L_2$ 정규화를 추가하면, 각 미니 배치에서 모든 파라미터에 대해 $L_2\;norm$을 계산해야해서 계산이 되지 않습니다. (저자들의 경우엔 bilions까지 사이즈 스케일링이 이루어졌다고 합니다..) {sparse feature들에서 업데이트할 필요 없는 0들도 다 업데이트 하니까 차원이 너무 커서 아마 안되는 듯합니다}
본 논문에선, 미니 배치에 나타나는 0이 아닌 feature들에 대해 $L_2\;norm$을 계산해 계산이 되게끔 하는! 새로운 mini-batch aware regularization을 개발합니다.
그리고 입력들의 분포와 관련된 recitified point들을 적응적으로 조정하여 흔히 사용되는 PReLU를 일반화하는 data adaptive activation function를 설계합니다! 이것은 sparse feature들을 갖는 산업 네트워크를 훈련하는데 도움이 될 것입니다.

<br/><br/><br/>

# 2. _Related work_
---

CTR 예측 모델의 구조는 얕은 것에서부터 깊은 것까지 발전해왔습니다. 동시에, CTR모델에 사용되는 특징들의 차원이나 샘플들(?)의 수는 점점 더 커져갔습니다. 
성능을 향상시키기 위해 더 좋은 특징 관계들(feature relations)를 추출하기 위해서, 몇몇 연구들은 모델 구조의 설계에 집중했습니다. 

[NNLM](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)는 language 모델링에서 차원의 저주를 피하기 위해 각 단어에 대한
distributed representation을 학습합니다. 보통 embedding으로 불리는 이 방법은 large-scale sparse input들을 다루는 많은 NLP 모델들과 CTR 예측 모델들에 영감을 주었습니다. 
{NNLM은 꼭 읽어봐야 할 것 같습니다. 이번 논문 다음으로 읽읍시다.}

[LS-PLM](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1704.05194.pdf)과 FM 모델은 하나의 hidden layer를 가진 네트워크 클래스로 볼 수 있습니다. 
이것은 sparse inputs에 embedding layer를 먼저 적용하고난 다음에 target fitting을 위해 특별히 설계된 transformation functions을 부과합니다. 동시에 특징들 사이의 combination relation을 포착하기 위해서 말이죠. 
{이게 논문을 안 읽어봐서 정확한 내용은 모르겠습니다.}

Deep Crossing, [Wide&Deep learning](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1606.07792.pdf) 그리고 [유튜브 추천 CTR 모델](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)은
LS-PLM과 FM의 transformation function을 complex MLP network로 바꿈으로써 확장시킵니다. complex MLP network로 바꾸는 것은 model capability를 굉장히 향상시킵니다! 
[PNN](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1611.00144.pdf)은 embedding layer 다음에 product layer를 넣어서 high-order feature interaction을 포착하려 했습니다.
{high-order라는건 고차원을 이야기하는건가?} 
[DeepFM](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1703.04247.pdf)은  feature engineering 없이 Wide&Deep에서 "wide" 모듈로서 factorization machine을 사용합니다.
전반적으로 위 방법들은 (sparse한 특징들의 dense representation을 학습하기 위한)embedding layer의 combination와 (특징들의 combination 관계들을 자동으로 학습하는)MLP을 갖는 유사한 모델 구조를 따릅니다.
**이런 CTR 예측 모델들은 사람이 직접 feature engineering을 하는 것을 굉장히 줄여줍니다!**  그러나 유저 행동들(user behaviors)이 많은 경우, 특징들은 종종 길이가 변하는 데이터를 가집니다. 
예를 들면, 검색된 용어 또는 유튜브 추천 시스템에서 시청된 비디오등이 있습니다. 이런 모델들은 종종 임베딩 벡터들의 일치하는 목록을 sum/average pooling을 통해서 고정된 길이의 벡터로 변환합니다. 
**단, 이 때 정보 손실을 야기하게 됩니다.** 
저자들이 제안하는 DIN은 주어진 광고들에 대한 representation 벡터들을 적응적으로 학습함으로써 이런 문제들을 해결합니다. 동시에 모델의 표현 능력또한 상승시키면서 말이죠!

Attention mechanism은 [NMT(Neural Machine Translation)](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1409.0473.pdf) 분야로부터 유래됩니다. 
NMT는 기대되는 annotation을 얻기 위해 모든 annotation의 가중치 합을 얻습니다. 그리고 다음 목표 단어 생성에 관련된 정보에만 오직 집중하죠. 
[Deep intent](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.kdd.org/kdd2016/papers/files/rfp0289-zhaiA.pdf)는 검색 광고의 맥락에 attention을 적용합니다. 
NMT와 유사하게 이것은 [RNN](https://ieeexplore.ieee.org/document/6795228)을 모델 텍스트에 사용하고 각각의 쿼리에 핵심 단어들에 집중하게끔 돕는 global hidden vector를 배웁니다. 
이것은 attention의 사용이 쿼리 또는 광고의 주요 의도를 포착하는데 도움이 될 수 있다고 보여줍니다. DIN은 관련된 유저 behavior들에 대한 soft-search를 하기 위한 local activation unit을 설계합니다.
그리고 DIN은 주어진 광고에 관한 유저 관심도들의 adaptive representation을 얻기 위해 weighted sum pooling을 씁니다.
user representation vector는 광고와 사용자간에 상호작용이 없는 DeepIntent와는 다르게, 다른 광고들에 따라 다릅니다. 

