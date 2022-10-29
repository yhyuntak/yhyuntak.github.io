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

# 3. _CANDIDATE GENERATION_
---



<br/><br/>

# 4. _Deep Interest Network_
---

Sponsored search와 다르게, 사용자들은 명시적으로 의사를 표현하지 않고 전시 광고 시스템에 들어옵니다. _{Sponsored search가 무엇인지 모르겠습니다}_
CTR 예측 모델을 만들 때, 많은 Historical behavior들로부터 사용자 관심사들을 추출하기 위한 효과적인 접근법이 필요합니다.
사용자들을 묘사하는 특징들과 광고들은 광고 시스템의 CTR 모델링에 있어 기본적인 요소들입니다.
**이런 특징들을 합리적으로 사용하고 정보를 추출하는 것은 중요합니다!**

<br/>

## 4.1. Feature Representation

실제 CTR 예측 task의 데이터는 다음 예제처럼 대부분 다중 그룹 카테고리형(multi-group categorial form)입니다.

* weekday = Friday
* gender = Female
* visited_cate_ids = {Bag,Book}
* ad_cate_id = Book

이 특징들은 인코딩[ [4](https://arxiv.org/pdf/1606.07792.pdf),[19](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/41159.pdf),21]을 통해 고차원의 sparse binary feature로 변환됩니다.
수학적으로 알아야할 개념들에 대해 적어두겠습니다.

* i번째 feature group의 encoding vector : $t_i\;\in\;R^{K_i}$. $K_i$은 feature group i의 차원이고, feature group i은 $K_i$ 유일한 광고들을 포함을 의미함. _{$K_i$ unique ads가 무엇인지 아직 모르겠습니다.}_
* $t_i\[j\]$ : $t_i$의 j번째 요소, 이것은 0 또는 1의 값을 가지며 $\Sigma_{j=1}^{K_i} t_i\[j\]=k$를 만족함.
* if $k=1$, one-hot encoding이고, else if $k>1$, multi-hot encoding이다. _{one-hot은 벡터 하나에 1이 한개인 것이고, multi-hot은 2개 이상인 것을 얘기합니다.}_
* 1개의 인스턴스 예시 : $x\;=\;[t_1^T,...,t_2^T,t_M^T]^T$. M은 feature 그룹들의 수이고 $\Sigma_{i=1}^{M} K_i=K$를 만족함. $K$는 전체 feature space의 차원.

이 방식으로 이번 세션 4.1의 시작에서 얘기했던 예제를 수학적으로 표현해보면 다음과 같습니다.

* weekday = Friday -> [0,0,0,0,1,0,0] _{아마 월,화,수,목,금,토,일 을 binary로 표현한 것 같네요.}_
* gender = Female -> [0,1] _{성별에는 남,여 밖에 없습니다.}_
* visited_cate_ids = {Bag,Book} -> [0, ... , 1, ... , 1, ... , 0] _{다른 항목들이 무엇이 더 있는지 모르는 상황이므로 Bag,Book에만 1이 되어있는 모습입니다.}_
* ad_cate_id = Book -> [0, ... , 1, ... , 0]  _{위 상황과 동일합니다.}_

저자들의 시스템에서 전체 feature set은 테이블 1과 같이 표현됩니다.

|![테이블 1](/assets/images/다양한 공부/논문/CTR/Deep_Interest_Network/테이블 1.png)|
|:--:|
|_테이블 1_|

보면 4개의 카테고리로 구성되어있고, user behavior features는 전형적인 multi-hot encoding 벡터들이고, 사용자 관심사들의 많은 정보를 포함합니다.
**여기서 저자들은 combination feature들을 사용하지 않았음을 알아야합니다. 저자들은 deep neural network로 특징들의 interaction을 포착할 것이기 때문에 그런 것 같네요!?**

<br/>

## 4.2. Base Model(Embedding&MLP)

|![basemodel](/assets/images/다양한 공부/논문/CTR/Deep_Interest_Network/basemodel.png)|
|:--:|
|_그림 2. Base model_|

가장 인기있는 모델 구조[3,4,21]는 그림 2처럼 유사한 Embedding&MLP paradigm을 가집니다. 저자들은 이 모델을 DIN의 base model로 사용한다고 하네요. 
저자들의 DIN의 구조는 그림 3과 같습니다.

|![DINmodel](/assets/images/다양한 공부/논문/CTR/Deep_Interest_Network/DIN.png)|
|:--:|
|_그림 3. Deep Interest Network_|

### Embedding layer

입력 데이터들이 고차원의 binary 벡터들이라서 이것을 저차원의 dense representation으로 변환하기 위해서 embedding layer를 사용한다고 합니다.
i번째 feature 그룹 $t_i$에 대해, i번째 embedding dictionary를 $W^i=[w_1^i,...,w_j^i,...,w_{K_i}^i]\;\in\;R^{D\times K_i}$라고 표현합니다.
$w_j^i\in R^D$는 D차원의 embedding vector라고 합니다. Embedding operation은 table lookup 메커니즘을 따른다는데.. 이게 뭔진 잘 모르겠네요.

* 만약 원-핫 벡터 $t_i$라면 j번째 요소가 $t_i[j]=1$가 됩니다. 그럼 $W^i$와 $t_i$가 곱해지면 $w_j^i$만 남습니다. 이걸 single embedding vector $e_i$라고 합니다
* 만약 멀티-핫 벡터 $t_i$라면 j번째 요소는 여전히 $t_i[j]=1$이지만, $j\in \{i_1,i_2,...,i_k\}$ k개의 1이 있을 겁니다. 그럼 $t_i$의 embedded representation은 
리스트처럼 뽑혀서 single embedding vector $e_{i_k}$들이 묶인 $\{ e_{i_1},e_{i_2},...,e_{i_k} \}=\{ w^i_{i_1},w^i_{i_2},...,w^i_{i_k}\}$가 됩니다.

### Pooling layer and Concat layer

사용자마다 행동의 수가 다릅니다. 그래서 멀티-핫 behavioral feature vector $t_i$의 0이 아닌 값의 수는 데이터마다 다르죠. 이것은 embedding 벡터들의 리스트의 길이를 다양하게 만듭니다.
fully connected network은 오직 고정된 길이의 입력들만 다룰 수 있기에, embedding 벡터들의 리스트를 pooling layer를 통해 고정된 길이의 벡터로 바꿔야합니다. 

$$
e_i=pooling(e_{i_1},e_{i_2},...,e_{i_k})
$$

가장 많이 사용되는 pooling layer들은 **sum pooling**과 average pooling입니다. 전자는 embedding vector들의 리스트를 element-wise sum을 하고, 후자는 평균을 계산하죠.

embedding과 pooling layer들은 원래의 sparse한 feature들을 여러개의 고정된 길이의 representation vector들로 매핑하는 group-wise 방식으로 작동합니다. {아마 함께 작동한다는 느낌인 것 같습니다.}
그러면 모든 벡터들은 전체적인 representation 벡터를 얻기위해 함께 concatenate됩니다. 

### MLP
앞서 concatanated된 dense representation 벡터가 주어지면, fully connected layer들은 feature들의 조합을 자동으로 학습하곤 합니다. 
최근에 발전된 방법[4,5,10]은 더 나은 정보 추출을 위해서 MLP의 구조를 설계했죠.

### Loss
base모델에 사용된 목적함수는 다음과 같은 negative log-likelihood 함수입니다.

$$
L=-\frac{1}{N}\sum_{(x,y)\in S} (y\log p(x) + (1-y)\log (1-p(x)))
$$

※ $S$는 N개의 학습 셋을 얘기하고, $x$는 입력 데이터, $y$는 0 또는 1을 갖는 이진 label입니다. 그리고 $p(x)$는 softmax layer를 통해 얻는 sample $x$가 클릭 될 것으로 예측되는 확률입니다.


<br/>

## 4.2. The structure of Deep Interest Network

테이블 1의 feature들에서, 사용자 행동 feature들은 중요하고 사용자 관심도를 모델링하는데 핵심 역할을 합니다.

Base model는 앞선 방법으로 사용자 관심도에 대한 고정된 길이의 representation vector를 얻습니다. 이 representation vector는 
어떤 후보 광고든 상관 없이 주어진 사용자에 대해 같음을 유지합니다. 
그래서 **제한된 dimension을 갖는 사용자 representation vector는 사용자의 다양한 관심사들을 표현하기 하는데에 bottleneck이 됩니다.**
이걸 해결하기 위한 간단한 방법은 embedding vector의 차원을 확장하는거지만.. 이렇게하면 learning parmameter의 크기가 매우 커질 것입니다.
그렇게 되면 제한된 학습 데이터에서 overfitting이 일어날 것이고, 용량과 계산량에 부담을 줘서 산업의 online system에선 쓸 수 없게되죠.

그럼 어떻게 이 문제를 해결해야할까요!? **사용자 관심사들의 local activation 특성**은 우리에게 Deep Interset Network를 만들 수 있게 영감을 줍니다!
챕터3에서 젊은 엄마의 예시를 들었는데, 기억하시나요? 그녀는 보여지는 새로운 핸드백을 찾고 클릭합니다. click action의 추진력을 해부해 봅시다! 
전시된 광고는 그녀의 과거의 행동들을 soft-searching하고 최근에 토트백과 가죽 핸드백과 비슷한 물건을 검색했던 것을 찾아서 젊은 엄마의 관련된 관심사들을 
hit! 합니다. **즉, 전시된 광고와 관련된 행동들은 click action에 매우 영향을 주죠.** 그래서 DIN은 주어진 광고에 관한 locally activated interest의 representation에 
주목하면서 이 과정을 시뮬레이션합니다. 모든 유저들의 다양한 관심도들을 같은 벡터로 표현하는 것 대신에, 
DIN은 후보 광고에 대한 과거 행동의 관련성을 고려하여 사용자 관심도의 representation 벡터를 적응적으로 계산합니다.

그림 3은 DIN의 구조를 나타냅니다. Base model과 비교하면 DIN은 local activation unit을 소개하고 다른 틀은 같습니다. 특히, activation unit들은 사용자 behavior feature들에 적용됩니다.
이것은 주어진 후보 광고 A에 대한 사용자 representation $\nu_U$을 적응적으로 계산하기 위해 weighted sum pooling을 수행합니다.

$$
\nu_U(A) = f(\nu_A,e_1,e_2,...,e_H) = \sum^H_{j=1}a(e_j,\nu_A)e_j =\sum^H_{j=1}w_je_j
$$

※ $\{e_1,e_2,...,e_H\}$는 사용자 U의 길이가 H인 embedding vector list를 의미하고, $\nu_A$는 광고 A의 embedding vector입니다.
이 방법으로 $\nu_U(A)$는 다른 광고들마다 바뀝니다. $a(\ast)$는 그림 3에서 Goods N Weight라고 써있는 activation weight를 출력하는 feed-forward 네트워크입니다.
**_그리고 $a(\ast)$는 이들의 out product를 더한다는데.. 이게 그림이랑 엮어서 봐도 무슨 소린지 아직 모르겠습니다._**

local activation unit은 NMT에서 개발된 attention method와 비슷한 아이디어입니다. 그러나 다른 점은 constraint $\sum_i w_i=1$이 
사용자 관심도의 intensity를 보존하는 것을 목표로 하면서 $\nu_U(A)$ 계산을 느슨하게 해준다는 겁니다. 이전의 방법에선 $a(\ast)$의 출력에 softmax로 normalization을 하는데,
이걸 대신해서 contraint를 거는 것 같네요. 

예를 들어봅시다. 한 사용자의 과거 행동들이 90%는 옷, 10%는 전자기기를 포함하면, 티셔츠와 핸드폰의 두 광고가 주어졌을 때, 티셔츠 광고는 clothes에 속한 과거 행동의 대부분에 작용해서
아마 $\nu_U$가 핸드폰보다 가장 큰 값을 가질 겁니다. 기존의 방법은 이런 resolution을 잃는다고 하네요. 

저자들은 LSTM도 써보려했지만, 그리 좋은 결과를 얻진 못했나봅니다.

<br/><br/>

# 5. _Training Techniques_
---

이번엔 알리바바에서 대용량의 데이터들을 가지고 학습할 때 도움이 되는 2개의 기술들에 대해 소개하네요.

<br/>

## 5.1. Mini-batch Aware Regularization


|![그림 4](/assets/images/다양한 공부/논문/CTR/Deep_Interest_Network/그림 4.png)|
|:--:|
|_그림 4_|

Overfitting은 큰 문제죠. 예를 들어, 테이블 1에서 사용자의 visited_goods_ids와 광고의 goods_id를 포함한 0.6 billion의 차원을 갖는 goods_ids의 feature같은 경우, 
모델의 성능을 regularization없이 첫 epoch만에 급격히 떨어뜨립니다. 그림 4에서 어두운 초록색 라인처럼 말이죠. 그렇다고 직접 전통적인 regularization을 적용해 버리는 것도 
실용적인 방법은 아닙니다(릿지,라쏘 같은 규제말이죠). 정말일까요?

한번 릿지 규제 $l_2\;regularization$을 예로 들어봅시다. 규제 없이 SGD를 베이스로 하는 최적화 방법에서는 오직 각각의 mini-batch에서 나타나는 0이 아닌 sparse feature들만 필요합니다.
그러나 릿지 규제를 추가하면, 각각의 mini-batch마다 모든 파라미터들(가중치 같은)에 대해 L2 norm을 계산해야합니다. 왜냐하면, 릿지 규제는 파라미터(가중치)의 절대값을 최대한 작게 만드는게 목표이기 때문이죠.
그래서 너무 많은 계산이 필요하게 되니까.. 문제가 발생합니다.  

이 논문에서 효과적인 mini-batch aware regularizer를 소개합니다. 이건 각각의 미니 배치에서 나타나는 sparse feature들의 파라미터들에 대해 L2 norm을 계산합니다. 
사실, embedding dictionary가 CTR 네트워크의 대부분의 파라미터들에 기여하고 많은 계산량의 어려움을 유발합니다. $W \in R^{D\times K}$는 전체 embedding dictionary의 파라미터들을 얘기합니다.
D는 임베딩 벡터의 차원이고, K는 feature space의 차원입니다. $W$에 대한 릿지 규제를 확장해봅시다.

$$
L_2(W) = || W ||_2^2 = \sum_{j=1}^K || w_j ||^2_2 = \sum_{(x,y)\in S} \sum_{j=1}^K \frac{I(x_j \neq 0)}{n_j} || w_j ||^2_2
$$

* $w_j \in R^D$ : j번째 임베딩 벡터
* $I(x_j \neq 0)$ : 데이터 객체 x가 feature id $j$를 갖는지 여부
* $n_j$ : 모든 샘플들에서 feature id $j$가 발생하는 횟수

{위 식처럼 되는 이유를 제 생각대로 설명해보겠습니다. K차원을 갖는 데이터 객체 $x$가 N개 있다고 한다면, $X\in R^{N\times K}$가 됩니다. 
데이터 객체 벡터 $x$는 원-핫/멀티-핫 인코딩으로 이루어져있기 때문에, j번째 요소에는 0또는 1이 있겠죠. 그럼 j번째 column에는 N개의 요소들이 있는데, 
그 중 1인 것들의 수를 세어 $n_j$를 만드는 겁니다. 그렇게 되면 위 식의 제일 마지막 항의 첫 summation과 I()와 $n_j$가 이해가 될겁니다. $x$의 j번째 요소가 0이 아닌 것은 
$n_j$만큼 있으니 상수항으로 곱해져도 결국 같은 식이 되는겁니다. }

위 식은 mini-batch를 이용한 식으로 다음과 같이 바꿀 수 있습니다.

$$
L_2(W) = \sum_{j=1}^K \sum_{m=1}^B \sum_{(x,y)\in B_m} \frac{I(x_j \neq 0)}{n_j} || w_j ||^2_2
$$

* $B$ : 미니 배치의 수
* $B_m$ : m번째 미니 배치

미니 배치 $B_m$에서 feature id j를 갖는 데이터가 하나라도 있는지 묻는 $\alpha_{mj}=\max_{(x,y)\in B_m} I(x_j \neq 0)$을 설정합시다. 아마 미니배치 $B_m$에서 
j번째 요소에 1인 값의 수를 표현하는 것 같네요?

그러면 위 식은 다음과 **근사화**할 수 있습니다. 

$$
L_2(W) \approx \sum_{j=1}^K \sum_{m=1}^B  \frac{\alpha_{mj}}{n_j} || w_j ||^2_2
$$

m번째 미니베치에서, feature j의 embedding weights에 대한 gradient는 다음과 같습니다.

$$

w_j \leftarrow w_j-\eta [\frac{1}{ | B_m | } \sum_{(x,y)\in B_m} \frac{\partial L(p(x),y)}{\partial w_j}+\lambda \frac{\alpha_{mj}}{n_j}w_j] 

$$

<br/>

## 5.2. Data Adaptive Activation Function

|![그림 3](\assets\images\다양한 공부\논문\CTR\Deep_Interest_Network\그림 3.png)|
|:--:|
|_그림 3_|

PReLU는 다음과 같습니다. 

$$
f(s) = \begin{cases} 
 s,\;\text{if}\;s>0 & \\
 \alpha s ,\; \text{if}\;s <= 0
\end{cases} = p(s) \cdot s + (1-p(s)) \cdot \alpha s 
$$

$s$는 활성화함수 $f(\cdot)$의 1차원 입력이고, $p(s)=I(s>0)$은 $f(s)$를 $s와 \alpha s$로 스위칭 해주는 indicator 함수입니다.
$\alpha$는 학습파라미터이구요. 그림 3의 왼쪽은 제어함수로써의 PReLU를 보여줍니다. PReLU는 0에서 급격한 변화를 보입니다. 이것은 
각각의 레이어의 입력들이 다른 분포를 따를 때, 적절하지 않을 것입니다. 이것을 고려해서, 저자들은 새로운 data adaptive activation function을 그림 3의 오른쪽처럼 설계하고
_**Dice**_라고 이름 짓습니다. 식은 다음과 같습니다.

$$
f(s) = p(s) \cdot s + (1-p(s)) \cdot \alpha s, \; p(s) = \frac{1}{1+e^{-\frac{s-E[s]}{\sqrt{\text{Var}[s]+\epsilon}}}}
$$

학습 페이즈에서, $E[s]$와 $\text{Var}[s]$ 는 각각의 미니 배치에서 입력의 평균과 분산입니다. 
테스트 페이즈에서,  $E[s]$와 $\text{Var}[s]$ 는 데이터에 대한 평균  $E[s]$와 $\text{Var}[s]$ 를 움직이면서..? 계산됩니다. { 이게 무슨소린진 아직 모르겠네요 }
$\epsilon$은 작은 상수로 $10^{-8}$로 잡았다고 합니다.

Dice는 PReLU의 일반화로 보여질 수 있습니다. 핵심은 입력 데이터의 분포에 따라 recitified point를 adaptively하게 조정한다는 것입니다. 게다가, Dice는 스위칭하는데 스무스하게 사용하구요.
$E[s]$와 $\text{Var}[s]$ 가 0일땐, Dice는 PReLu에서 degenerate하다고 합니다..?

<br/><br/>

# 6. _Experiments_
---

{이제 실험파트입니다! 결과가 궁금하네요!}

<br/>

## 6.1. Datasets and Experimental Setup

데이터 셋은 다음의 3가지를 사용합니다.

* Amazon dataset
* MovieLens dataset
* Alibaba dataset

데이터 셋의 통계는 테이블 2와 같습니다. 자세한 설명은 논문을 참고해주세요.

|![테이블 2](\assets\images\다양한 공부\논문\CTR\Deep_Interest_Network\테이블 2.png)|
|:--:|
|_테이블 2_|

<br/>


## 6.2. Competitors

비교에 사용되는 모델들은 다음의 5가지 입니다.

*[LR](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/41159.pdf) : CTR 예측 태스크를 위한 딥 네트워크들이 있기 전에 널리 사용되던 얕은 모델
* BaseModel : 4.2에서 소개했던 Embedding&MLP 구조를 갖는 모델입니다. 
* [Wide&Deep](https://arxiv.org/abs/1606.07792) : 실제 산업 현장에서 사용되는 모델입니다. 자세한 내용은 논문을 참고합시다.
* [PNN](https://arxiv.org/pdf/1611.00144.pdf) : BaseModel에서 조금 향상된 모델입니다.
* [DeepFM](https://arxiv.org/pdf/1703.04247.pdf) : Wide&Deep에 "wide" 모듈로써 feature engineering job을 저장하는 factorization machine을 넣은 모델입니다.

<br/>


## 6.3. Metrics

평가 지표로 user weighted AUC를 사용합니다.

$$
AUC = \frac{\sum_{i=1}^n #\text{impression}_i \times AUC_i}{\sum_{i=1}^n #\text{impression}_i}
$$

그리고 모델들에 대한 상대적인 향상치를 측정하기 위해 RelaImpr 평가 지표를 도입합니다. 

$$
\text{RelaImpr} = (\frac{AUC(measured model)-0.5}{AUC(BaseModel)-0.5}-1)\times 100%
$$

여기서 BaseModel의 정확도를 기준으로 삼는걸 볼 수 있네요!

<br/>

## 6.4. Result from model comparison on Amazon dataset dan MovieLens dataset

|![테이블 3](\assets\images\다양한 공부\논문\CTR\Deep_Interest_Network\테이블 3.png)|
|:--:|
|_테이블 3_|

테이블 3은 각 데이터 셋들에 대한 결과를 보여줍니다. 각 실험들은 5번 반복해서 평균낸 것입니다. 
DIN이 가장 성능이 좋지만, 특히 사용자 행동 데이터가 굉장히 많은 아마존데이터셋에서 엄청난 성능을 보여줍니다!
저자들은 이 결과의 공을 local activation unit structure로 돌리네요. 

<br/>

## 6.5. Performance of reqularization

|![그림 4](\assets\images\다양한 공부\논문\CTR\Deep_Interest_Network\그림 4.png)|
|:--:|
|_그림 4_|

|![테이블 4](\assets\images\다양한 공부\논문\CTR\Deep_Interest_Network\테이블 4.png)|
|:--:|
|_테이블 4_|

사실, MovieLens와 아마존 데이터셋의 feature의 차원이 그리 크지 않아서 저자들이 생각하던 overfitting 문제를 제대로 만나지 못했다고 합니다.
알리바바의 데이터셋은 엄청 차원이 커서, overfitting문제를 만났다고 하네요. 어떤 규제도 걸지 않고 overfitting문제를 만나게 되면, 그림 4의 어두운 초록색 선처럼
epoch 1부터 모델의 성능이 급격히 떨어집니다. 그래서 저자들은 다음의 4가지의 규제 방식으로 각각 실험해서 차이를 확인해보려고합니다.

* [Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) 
* Filter
* Regularization in [DiFacto](https://www.cs.cmu.edu/~muli/file/difacto.pdf) : feature들과 자주 관련되는 파라미터들의 지나친 규제를 줄입니다.?
* MBA : 저자들이 제안하는 방법입니다.

그림 4와 테이블 4는 결과를 보여줍니다. 

* Dropout : overfitting을 꽤 막아주지만, 수렴이 느리군요.
* Filter : Dropout과 비슷해 보입니다.
* DiFacto : 이것은 goods_id가 높은 빈도 수를 보이기에, 패널티를 주었습니다. 그래서 filter보다 더 안좋은 것 같네요? 

저자들의 방법은 다른 방법들에 비해 상당히 overfitting 문제를 막아줍니다. 
