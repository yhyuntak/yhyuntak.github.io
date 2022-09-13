---
title: "논문 공부: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications - MobileNet V1"
excerpt : MobileNet 공부
categories:
  - 컴퓨터 비전
  - 논문 리뷰
toc: true

---

본 글은 [논문 MobileNet V1](https://arxiv.org/abs/1704.04861)을 읽고 공부하는 글입니다.

---

# 0. _Abstract_
모바일넷은 간결한 구조를 기반으로 하기위해 depthwise separable convolution을 사용해 light weight deep neural network를 만들었다. 
저자들은 latency와 accuracy사이의 trade off를 효과적이게 하는 2개의 간단한 global hyperparameter를 소개한다. 
이 파라미터들은 model builder가 problem의 제약들에 기반한 자신의 애플리케이션에 적합한 크기의 모델을 선택할 수 있게 해준다.

<br/>

---

# 1. _Introduction_

시간이 흐르면서 높은 정확도를 갖는 복잡하고 deep한 네트워크들이 많이 나왔고 트렌드였다.
그러나 이런 정확도를 향상시키는 네트워크들은 size나 speed에 관해 좀 더 효과적으로 발전시키지 못하였다.
로보틱스, 자율주행차, 증강현실 등 recognition task들은 계산력이 제한된 platform에서 제때 실행되어야 한다.
즉, 네트워크 구조들은 효과적이게 작고 좋은 성능을 보여야한다.

Section 2는 작은 model을 만들기 위한 사전 작업들에 대해 이야기할 것이고,
Section 3는 본 네트워크의 구조와 2개의 하이퍼 파라미터들 **_width multiplier_** 와 **_resolution multiplier_**를 이야기 할 것이다. 이 파라미터들은 네트워크를 보다 더 작고 효율적이게 만든다.
Section 4는 실험 결과들에 대해 이야기할 것이고
Section 5는 요약 및 결론을 이야기할 것이다.

<br/>

---

# 2. _Prior work_


이런 효율적인 네트워크를 만들기 위한 많은 접근들은 pretrained network를 압축하거나, 작은 네트워크를 직접적으로 학습시키는 것으로 카테고리화 되었다.
이 논문은 model developer가 그들의 어플리케이션에 대한 (latency, size와 같은) resource restriction들을 충족시키기 위한 작은 네트워크를 특별히 선택할 수 있도록 하는 네트워크 아키텍처의 종류를 제안한다.
모바일넷은 걸리는 시간(latency)를 최적화 하는데 초점을 맞추지만, 작은 네트워크들 또한 만들어낸다.
**많은 논문들이 size만 신경쓰지 속도에 대해선 신경쓰지 않는 점과 다르다.**

모바일넷은 [26]번 논문에서 처음으로 소개된 depthwise separable convolution으로 부터 주로 만들어지고, 그 다음에, 첫번째의 적은 수의 레이어들에서 계산량을 줄이기 위해 [13]번 논문의 Inception model이 사용 됬다.
Flattend networks[16]은 convolution들을 분해한 네트워크를 만들고, 분해된 네트워크들의 잠재력을 보여줬다.
현재 이 논문과 독립적으로, Factorized Networks[34]는 topological connection들의 사용 뿐 아니라, 유사한 factorized convolution을 소개한다.
다음으로, Xception network [3]은 depthwise separable filter들이 inception V3 networks를 능가하기 위해 확장되는지 증명했다.
다른 small network는 Squeezenet [12]이고 이는 아주 작은 small network를 디자인하기위해 bottleneck을 사용한다.
다른 계산량을 줄이는 네트워크들은 structured transform networks [28], deep fried convnets [37]이 예시이다.

small networks를 얻기 위한 다른 접근은 pretrained network들을 압축하거나 분해하거나 줄이는 방식으로 진행했다.
literature에서 Compression based on product quantization [36], hashing [2], and pruning, vector quantization and Huffman coding [5] have been proposed .
추가적으로 다양한 factorization들은 pretrained networks [14,20]의 속도를 높이기 위해 제안됬다.
small networks를 training하기 위한 방법은 distillation [9]이고 이는 smaller network를 교육하기 위한 larger netowkr를 사용하는 것이다.

<br/>

---

# 3. _MobileNet Architecture_

이제 MobileNet이 만들어지는 핵심 레이어들 (depthwise separable filter)에 대해 이야기할 것이다.

## 3.1. _Depthwise Separable Convolution_

depth wise separable convolution은 standard convolution을 depthwise convolution과 pointwise convolution이라 불리는 1X1 convolution으로 factorize하는 걸 이야기한다.
depthwise convolution은 각각의 입력 채널에 single filter를 적용한다.
그런 다음에 pointwise convolution은 1X1 convolution을 사용해 outputs의 depthwise convolution을 결합한다.
기존의 convolution은 필터와 입력값을 결합해 새로운 output set을 내놓는다.
depthwise separable convolution은 이 과정을 2개의 layer들로 쪼갠다.
하나는 filtering을 위한 separate layer 이고, 다른 하나는 combining을 위한 separate layer이다.
이 factorization은 엄청나게 모델의 크기와 계산량을 줄일 수 있다.

| ![예제](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/그림1.png) | 
|:--:| 
| *그림 1* |

Figure 2.(a)는 standard convolution을 나타내는데, 이 것은 앞서 말한 것과 같이 2.(b) depthwise convolution, 2.(c) 1X1 pointwise convolution 으로 나뉜다.
기존의 convolutional layer는 입력으로써 $$D_F \times D_F \times M$$ 크기를 갖는 feature map F를 갖고 $$D_F \times D_F \times N$$ 크기의 feature map G를 만든다. $$D_F$$ 는 입력 feature map가 정사각형이라 할 때, width=height를 나타낸다.
이 뜻의 해석은 다음과 같다. 만약 입력 이미지의 크기가 (3,64,64)=(채널,width,height)이고 kernel의 size가 3x3 이고 채널의 수가 100, 그리고 stride = 1, padding size = 0이면,

![예제](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/식그림1.png)

위 계산에 의해서 아웃풋 이미지의 크기는 width = height = 64가 된다.
kernel의 size는 1x1이고 channel이 100이면, 입력과 kernel이 어떻게 연산되고 아웃풋의 채널의 수는 어떻게 될까??

**이런 식으로 생각하면 된다.**

convolution 연산시 입력 이미지가 (3,64,64)=(채널,width,height) 라면, kernel은 (3,1,1)=(채널,width,height) 인 놈이 100개가 있다고 생각하면 된다.
따라서 100개의 kernel이 각각 입력 이미지와 convolution하여 하나의 Output 채널을 형성하게 되고 이렇게 100개가 쌓여서 (100,64,64)가 되는 것이다.
즉 convolution 연산시 크기를 다시 써보자면 Input image(3,64,64)=(input의 채널,width of input,height of input) (*) kernel(3,100,1,1)=(input의 채널의 수 ,kernel의 개수,width of kernel,height of kernel) = Output image(100,64,64)=(output의 채널,width of output,height of output)

_※ (*)는 convolution을 뜻한다_

_※ 일반적으로 input image를 input feature map이라고도 부르고 output을 output feature map이라고도 부른다._


자, 이런 방식을 통해서 논문을 생각해보면 입력 이미지 or input feature map (M, DF, DF)=(input의 채널,width of input,height of input)가 들어왔고,
kernel(=filter) (M,N,1,1)=(input의 채널의 수 ,kernel의 개수,width of kernel,height of kernel) 으로 convolution을 할꺼다.
그렇게 되면 Output feature map은 (N, DG, DG)=(output의 채널,width of output,height of output) 이 된다. 라는 뜻이다.

기존의 convolutions의 computational cost는 다음과 같이 정의된다. (이유를 잘 모르겠다. 파라미터의 개수를 이야기하는 것이 아닌 것 같다.)

![예제](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/식그림2.png)

Mobilenet은 이런 term들과 interactions을 설명한다. 

먼저 MobileNet은 Dk와 N의 interaction을 멈추기 위해 depthwise separable convolutions를 사용한다.
기존의 convolution operation은 convolutional kernel들에 기반한 feature들을 filtering하고 새로운 표현을 만들기 위해 feature들을 combining하는 효과를 가진다.
filtering하고 combination 단계들은 depthwise separable convolution으로 위에서 언급한 computational cost를 상당히 줄이는 작업을 사용하면서 2단계로 나뉘게 된다.

Depthwise separable convolution(DSC)는 2개의 layer들로 구성되어 있다.
1. depthwise convolutions
2. pointwise convolutions
각 input channel 당 single filter를 적용하기 위해 depthwise convolution을 사용한다.
Pointwise convolution 는 1 x 1 convolution이다. 이것은 depthwise layer의 output의 linear combination을 만들기 위해 쓰인다. 그리고 MobileNet은 batchnorm과 ReLU를 두 convolution에 이용한다는 듯.
Depthwise convolution은 input channel당 1개의 filter를 쓰는데, 그 식은 아래와 같이 쓸 수 있다.

![예제](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/식그림3.png)


위 수식은 그냥 input에 kernel을 취한 것이 output feature map이 된다는 것을 식으로 표현한 것이다.
그림으로 나타내면 다음과 같다.

| ![예제](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/그림2.gif) | 
|:--:| 
| *그림 2* |

그림 2.는 Input image(=input feature map)에 padding을 씌우고 kernel(=filter)를 사용해 convolution을 하는 모습을 나타낸 것이다.
그림 2.에서 kernel의 중심과 input image가 겹치는 부분을 보라색 칸으로 표현했다.
여기서 stride = 1이라면, 보라색 칸이 input image를 모두 훑으면서 convolution이 진행될 것이다.
그렇게 되면 kernel이 DF X DF 번 Input image와 padding을 훑으면서 연산이 될 것이다.

그림 2.는 channel이 하나인 것을 표현했는데, 아래 그림 3.은 M개의 채널을 가진 Input image에 대해 depthwise convolution을 나타낸다.

| ![그림3](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/그림3.gif)|
|:--:|
| *그림 3* |

그림 3.은 그림 2.를 사용하여 depthwise convolution을 하는 모습이다.
그림 2.를 사용한다고 해서 기존의 convolution과 다른 연산을 하는 것은 아니다.
단지 입력으로 들어오는 image (= Input feature map)의 channel (= depth)의 각각에 ( DF X DF X ⅰth ) single filter ( DK X DK X ⅰth )를 사용해 ( ⅰ= 1,2, ... , M)
같은 channel (= depth)의 Output of depthwise convolution ( DDepth X DDepth X M )을 생성하는 것 뿐이다.


다음으로 Pointwise conovolution에 대해 설명하겠다. 아래 그림 4.을 보자.

| ![그림4](/assets/images/다양한 공부/논문/컴퓨터 비전/MobileNets/그림 4.gif)|
|:--:|
| *그림 4* |


그림 4.는 Pointwise convolution 과정을 나타낸다.
방식은 다음과 같다.
Output of depthwise convolution ( DDepth X DDepth X M ) 을
Kernel ( 1 X 1 X M X N ) 를 사용하여 Convolution을 진행한다.
다시 말하자면, Output of depthwise convolution을
ⅰ번째 Kernel ( 1 X 1 X M ) 을 사용해 (ⅰ= 1, 2, ... , N ) Convolution 하는 것을 1st 부터 Nth 까지
반복해 N channel을 갖는 Output image( = Output feature map)을 생성한다.


**그러면 이런 과정을 굳이 거치는 이유가 무엇인가 ??
그 답은 연산량에 있다.**


기존의 Input image ( DF X DF X M )을 Kernel (DK X DK X M X N) 으로 연산을 하게되면,
< 이 때, DK = 3, padding = 1 , stride = 1 로 설정한다 >
DK X DK X M 의 크기를 갖는 Kernel은 DF X DF 번 연산을 하게 되고,
DK X DK X M 의 크기를 갖는 Kernel은 N개 가지고 있으므로
이런 과정을 총 N번 진행하므로 DK X DK X M X DF X DF X N 의 연산량을 가진다.
**이는 논문에서 (2)번 식을 의미한다.**

**하지만 Depthwise separable convolution을 사용하면 연산량이 어떻게 변할까?**


먼저 Depthwise convolution을 진행하게 되면,
기존의 Input image ( DF X DF X M )의 각 Channel을 Kernel (DK X DK X 1) 으로 연산을 하게되면,
< 이 때, DK = 3, padding = 1 , stride = 1 로 설정한다 >
Kernel (DK X DK X 1)이 DF X DF 번 연산을 하는데,
Input image가 M Channel을 가지므로,
Depthwise convolution은 DK X DK X DF X DF X M 의 연산량을 가진다.


다음으로 Pointwise convolution을 진행하게 되면,
Output of depthwise convolution ( DDepth X DDepth X M ) 을
Kernel ( 1 X 1 X M X N ) 를 사용하여 Convolution을 진행하면,
< 이 때, DK = 3, stride = 1 로 설정한다 >
Kernel ( 1 X 1 X M )이 Output of depthwise convolution ( DDepth X DDepth X M ) 과의
Convolution 연산을 총 N ( kernel의 개수 ) 개 진행하게 되므로
Pointwise convolution은 DDepth X DDepth X M X N 의 연산량을 가진다.
이제 기존의 연산량과 Depthwise separable convolution의 연산량을 비교해보자.
기존의 연산량은 DK X DK X M X DF X DF X N 이다.
Depthwise separable convolution의 연산량은
DK X DK X DF X DF X M + DDepth X DDepth X M X N 이다.
이 때, depthwise convolution을 DK = 3, padding = 1 , stride = 1 으로 설정하고 진행하면
DDepth 는 DF 와 같다.

따라서
DK X DK X DF X DF X M + DDepth X DDepth X M X N
= DK X DK X DF X DF X M + DF X DF X M X N
= DF X DF X M X ( DK X DK + N )
이 된다.


감이 오지 않을 수 있다. 그러면 한번 값을 대입해보자.
DK = 3 , DF = 128, M = 64, N = 128 이라고 하면,
기존의 연산량은 3*3*64*128*128*128 = 1,207,959,552 이고 약 12억이고
Depthwise separable convolution의 연산량은 128*128*64*(3*3+128) = 143,654,912으로 약 1억 4천.
약 10분의 1이 줄어든 셈이다. 10분의 1이 줄어들었다는게 되게 얼마 안되는거 같지만
12억에서 1억4천으로 줄었다는 것만 보면 **엄청난 계산량의 감소폭을 직관적으로** 알 수 있다.


<br/>

---

# 1. _Introduction_


<br/>

---

# 1. _Introduction_

시간이 흐르면서 높은 정확도를 갖는 복잡하고 deep한 네트워크들이 많이 나왔고 트렌드였다.
그러나 이런 정확도를 향상시키는 네트워크들은 size나 speed에 관해 좀 더 효과적으로 발전시키지 못하였다.
로보틱스, 자율주행차, 증강현실 등 recognition task들은 계산력이 제한된 platform에서 제때 실행되어야 한다.
즉, 네트워크 구조들은 효과적이게 작고 좋은 성능을 보여야한다.

Section 2는 작은 model을 만들기 위한 사전의 work에 대해 이야기할 것이고,
Section 3는 본 네트워크의 구조와 2개의 하이퍼 파라미터들 <width multiplier> 와 <resolution multiplier>를 이야기 할 것이다. 이 파라미터들은 네트워크를 보다 더 작고 효율적이게 만든다.
Section 4는 실험 결과들에 대해 이야기할 것이고
Section 5는 요약 및 결론을 이야기할 것이다.

<br/>

---

# 1. _Introduction_

시간이 흐르면서 높은 정확도를 갖는 복잡하고 deep한 네트워크들이 많이 나왔고 트렌드였다.
그러나 이런 정확도를 향상시키는 네트워크들은 size나 speed에 관해 좀 더 효과적으로 발전시키지 못하였다.
로보틱스, 자율주행차, 증강현실 등 recognition task들은 계산력이 제한된 platform에서 제때 실행되어야 한다.
즉, 네트워크 구조들은 효과적이게 작고 좋은 성능을 보여야한다.

Section 2는 작은 model을 만들기 위한 사전의 work에 대해 이야기할 것이고,
Section 3는 본 네트워크의 구조와 2개의 하이퍼 파라미터들 <width multiplier> 와 <resolution multiplier>를 이야기 할 것이다. 이 파라미터들은 네트워크를 보다 더 작고 효율적이게 만든다.
Section 4는 실험 결과들에 대해 이야기할 것이고
Section 5는 요약 및 결론을 이야기할 것이다.

<br/>

---
