---
title: "논문 공부: DenseFusion: 6D Object Pose Estimation by Iterative Dense Fusion"
excerpt : DenseFusion 공부
categories:
  - 컴퓨터 비전
  - 논문 리뷰
toc: true

---

본 글은 [논문 DenseFusion](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1901.04780.pdf)을 읽고 공부하는 글입니다.

---

# 0. _Abstract_
---

RGB-D이미지로 부터 6D object pose estimation을 수행하는 중요한 기술적 어려움은 2개의 보완적인 데이터 소스를 완전히 활용하는 것이다.
선행 작업에서 RGB이미지와 depth 이미지로부터 따로 정보를 추출하거나 다음 단계에서 사용하는 것은 매우 clutter한 장면과 실시간 apllications에서 그들의 성능을 제한한다.
저자들은 이런 작업에서 DenseFusion을 제시한다. 
DenseFusion은 2개의 데이터 소스들을 따로 처리하고 pixel-wise dense feature embedding을 추출하기 위한 참신한 dense fusion network를 사용한다.
그리고 이것으로부터, Pose를 예측한다.
게다가 저자들은 실시간에 가까운 inference를 달성하면서 pose estimation을 좀 더 향상시키는 end-to-end iterative pose refinement procedure을 통합한다.

<br/><br/><br/>

# 1. _Introduction_
---

6D object pose estimation(6D-OPE)은 real-world application에서 되게 중요한 것이다.
예를 들면 robotic grasping과 manipulation [7,34,43] 이라던지, 자율주행 [6,11,41] 등에 쓰이기 때문이다.
이상적으로, 6D-OPE의 solution은 전에는 RGB 이미지로만 이를 판단해야해서 다양한 shape,texture의 물체들을 다루고, 많이 어지러진 상황, 물체들이 겹쳐져있는 상황이라던지, 노이즈, 조도 등에 대해서도 robustness를 보이기 힘들었나 보다. 특히, 실시간으로 빠르게 처리하면서 말이다.
그런데 이제 값싼 RGB-D 센서가 나오면서 low-texutred objects라던지, 빛이 굉장히 약한 환경에서도 꽤 좋은 pose estimation이 이루어지는 듯.
그럼에도 불구하고, 여태 존재하던 방법들은 빠른 추론과 정확한 pose estimation (자세 추정)을 동시에 하는데 만족감을 주지 못했다.

**RGB-D를 사용한 여러 논문들에 대해 이야기함**

그러나, 이런 방법들 중 이제, PoseCNN에 같은 경우는 3D 정보를 완전히 이용하기 위해 정교한 post-hoc refinement 단계가 필요하다. 예를들어, ICP 과정[2] 같은 것 말이다 .
이런 refinement 단계는 각각 마지막 목표를 최적화할 수 없었고 느렸다.

**자율 주행에서 쓰이는 방법들에 대해 이야기함**

그러나 저자들이 입증하듯, 이런 방법들은 매우 겹쳐있는 것들이라던지에 대해 약하다.
이제 저자들이 제안하는 것에 대한 이야기가 나온다.
저자들은 RGB-D 입력들로부터 알고있는 object들의 6D-OPE를 하기위해 end-to-end 딥러닝 접근을 제안한다.
위 접근의 핵심은 RGB 값들과 point clouds를 픽셀 레벨 단위로 융합시키고 embed하는 것이다.
이 픽셀 단위의 fusion 계획은 저자들의 모델이 heavy occlusion을 필수적으로 다룰 수 있는 local appearance와 geometry information에 대해 명백하게 추론할 수 있게 해준다.
게다가 저자들은 end-to-end learning framework안에서 pose refinement를 수행하는 iterative method를 제안한다. 이는 모델의 성능을 실시간의 속도로 추론하면서 엄청나게 향상시켜준다.

<br/><br/><br/>

# 2. _Related Work_
---

## Pose from RGB images

**여러 논문들에 대한 내용에 대해 열거함.**

저자들의 방법은 이미지와 3D data를 활용해 end-to-end architecture 에서 3D로 object pose를 예측할 수 있다.

## Pose from depth / point cloud

**여러 논문들에 대한 내용에 대해 열거함.**

## Pose from RGB-D data

**여러 논문들에 대한 내용에 대해 열거함.**

저자들의 방법은 input space의 geometric structure을 유지하면서 2D appearance feature와 3D data를 융합한다. 그리고 저자들은 후처리 과정 없이 YCB-Video dataset에서 월등한 성능을 보인다.

<br/><br/><br/>

# 3. _Model_
---

저자들의 목표는 cluttered scene RGB-D 이미지에서 object들의 6D-OPE을 예측하는 것이다.
generality의 손실 없이, 저자들은 6D poses를 _homogeneous trasformation matrix p_ 로 표현한다.

$$p \in SE(3) $$

즉, 6D pose는 rotation R 과 translation t를 사용해 $p = \[R \| t\]$로 표현한다.

$$ R \in SO(3), \; t \in R^3 $$

저자들은 카메라 이미지들로부터 object들의 6D pose를 예측하므로, pose들은 camera coordinate frame에 대해 정의된다.

여러 악조건 ( ex) heavy occlusion, poor lighting, ...)에서 object의 pose를 찾는 것은 color와 depth 이미지의 channel들을 포함하는 정보를 결합함으로써 가능해진다.
**그러나 2개의 데이터 소스들은 서로 다른 space에 존재한다. 
여러 다른 종류들로 이뤄진 데이터 소스들로부터 특징을 추출하는 것과 그들을 적절히 합치는 것은 이 domain에서 중요한 기술적 어려움이 된다.**


저자들은 문제를 다음과 같이 해결한다.

1. Section 3.3의 다른 종류들로 이뤄진(heterogeneous) architecture : color와 depth 정보를 다르게 처리하여 각각의 데이터 소스의 native structure를 유지
2. Section 3.4의 dense pixel-wise fusion network : 데이터 소스들 간의 insrinsic mapping을 이용하여 color-depth fusion을 수행함
3. Section 3.6의 differentiable iterative refinement module를 좀 더 개선한 pose estimation

저자들의 refinement module은 공동으로 main architecture와 함께 훈련 될 수 있고 오직 전체 inference time에서 적은 시간만이 걸릴 것이다.

<br/>

## 3.1 _Architecture Overview_


|![네트워크 구조](\assets\images\다양한 공부\논문\컴퓨터 비전\DenseFusion\네트워크구조.png)|
|:--:|
|_그림 1. 네트워크 구조 및 1단계와 2단계의 1,2,3을 보여줌._|

그림 1은 전체적인 architecture(구조)를 보여준다.
구조는 2개의 주요 단계를 포함한다.

### 1단계 : dataset 설정

첫번째 단계는 color image를 입력으로써 받고, 이미 pre-trained된 네트워크로 semantic segmentation을 수행한다.
저자들은 각각 segment된 object들(objs)에 대해 다음을 수행한다.

1. 마스크된(segment된) depth pixels을 3D point cloud로 변환하여 얻는다
2. segment된 것에 해당하는 image 영역(patch)을 crop하여 2단계로 bounding box를 보낸다.

### 2단계 : 네트워크 학습

|![2단계](\assets\images\다양한 공부\논문\컴퓨터 비전\DenseFusion\2단계.png)|
|:--:|
|_그림 2. 2단계의 4를 보여줌._|

두번째 단계는 segmentation의 결과들을 처리하고 objs의 6D-OPE를 예측하는 단계이다. 총 4개의 단계로 구성되어 있다.

1. fully convolutional network : crop된 이미지에서 color information을 처리하고 각 픽셀을 color feature embedding으로 매핑한다.
2. PointNet-based [23] network : mask된 3D point cloud에서 각각의 point들을 geometric feature embedding으로 처리한다.
3. a pixel-wise fusion network : a), b)에서 얻은 embedding들을 결합하고, unsupervised confidence scoring에 기반하여 6D-OPE를 예측한다.
4. iterative self-refinement methodology : nework를 curriculum learning manner로 학습하고 반복적으로 estimation result를 개선한다.

<br/>

## 3.2 _Semantic Segmentation_


이 세션은 3.1에서 언급했던 첫번째 단계에 해당한다. 첫번째 단계는 이미지에서 물체의 관심있는 부분을 세그멘트한다.
저자들의 semantic segmentation network는 이미지를 input으로 받고 N+1 channelled semantic segmentation map을 만드는 encoder-decoder architecutre이다.
각각의 채널들은 활성화된 픽셀들이 N개의 가능한 known classes의 각각의 objs를 나타내는 binary mask 이다.
이 작업의 핵심은 pose estimation algorithm을 개발하는 것이지 segment network를 만드는 게 아니기 때문에, PoseCNN[40]의 segmentation architecture를 가져와서 사용했다.

<br/>

## 3.3 _Dense Feature Extraction_

domain에서 핵심적인 기술적 어려움은 color와 depth channels 로부터 얻은 정보의 올바른 추출과 이 두 것들의 반응이 좋은(synergistic) fusion 즉, 효과적인 융합이다.
color와 depth가 RGB-D frame에서 유사한 format을 제공할지라도, 이들의 정보는 서로 다른 spaces에 존재한다.
그러므로 저자들은 데이터 소스들의 inrinsic structure를 유지하는 embedding space로부터 color와 geometric features를 각각 만들어내기 위해 color와 depth를 처리한다.

### _Dense 3D point cloud feature embedding_

이전의 접근들은 depth image를 추가적인 image channel로써 처리하기 위해 CNN을 사용했다[16]. 
그러나, [16]은 depth channel의 intrinsic 3D structure를 소홀히 했다. 
대신에, 저자들은 알고있는 camera intrinsic들을 사용하면서 segment된 depth pixel들을 3D point cloud로 변환했고 geometric feature들을 추출하기 위해 PointNet 같은 구조를 사용했다.

PointNet [23]은 무질서한 point set들을 처리하면서 permutation invariance를 달성하기 위해 max-pooling같은 symmetric fuction을 사용한 선구자였다.
original 구조는 입력을 raw point cloud으로 받고 각 포인트의 주변과 point cloud 전체에 대한 정보를 encode하는 것을 배운다.
이 architecture에서 feature들은 shape classification, segmentation [23] 그리고 pose estimation [22,41]에서 효과적인 모습을 보인다.
저자들은 geometric embedding network를 제안한다.
이 네트워크는 segment된 점들 P의 각각을 dgeo-dimensional feature space로 매핑하면서 point당 dense feature를 만들어낸다.
저자들은 흔히 symmetric reduction function으로 사용되는 max-pooling을 대신해 average-pooling을 사용해 PointNet 구조에서 변형을 주었다. ( 그림 1에서 제일 오른쪽 중간에 average pooling 부분인듯)


### _Dense color image feature embedding_

color embedding network의 목표는 3D point feature들과 image feaure들 간의 dense한 대응(correspondences)를 형성하도록 픽셀 당 feature들을 추출하는 것이다.
이런 dense correspondences를 형성해야하는 이유는 다음 Section에서 명백해질 것이다.
image embedding network는 image의 크기를 $H \times W \times 3$ 에서 $H \times W \times d_{rgb}$ 로 매핑하는 CNN 기반의 encoder-decoder architecture 이다. 
embedding의 각각의 픽셀은 corresponding location에서 input image의 appearance information을 표현하는 drgb-dimensional vector이다.

<br/>

## 3.4 _Pixel-wise Dense Fusion_

지금까지 우리는 image와 3D point cloud input으로부터 dense feature들을 얻었고 이제 이 정보를 fuse, 즉 정보를 합쳐야한다!!
일반적인 접근은 segmented area로부터 얻은 dense color와 depth feature들로부터 global feature를 만들기 위한 것일 것이다.
그러나 segmentation의 에러나 heavy occlusion때문에, 이전 단계로부터 얻은 feature들의 set은 다른 objs나 배경의 일부분들의 point들 혹은 pixel들의 feature들을 포함할 지도 모른다.
그러므로 무턱대고 color나 geometric features를 globally하게 fusing하는 것은 estimation의 성능을 낮추게 될 것이다.
이제 저자들은 추출한 feature들, 특히 pose estimation을 위해 heavy occlusion과 불완전한 segmentation으로부터 추출한 feature들을 효과적으로 결합하는 pixel-wise dense fusion network를 제안한다.

### _Pixel-wise dense fusion_ 

저자들의 dense fusion network의 핵심 아이디어는 각각의 fused feature들에 기반하여 prediction을 만들기 위해 global fusion 대신 픽셀당 local fusion을 수행하는 것이다.
이 방법은, 잠재적으로 obj의 보이는 부분에 기반해 prediction들을 선택할 수 있고, occlusion과 segmentation noise의 효과를 줄일 수 있다.
구체적으로 저자들의.. < 이제부터 we가 나오면 그냥 저자나 우리들이나 섞어서 쓴다. > 어쨌든, 우리들의 dense fusion 과정은 먼저 camera intrinsic parameter들을 사용해서 image plane에 투영된 것을 바탕으로 각 포인트의 geometric feature를 이 feature들과 상응하는 image feature pixel과 연결한다.
feature들의 얻은 짝들 ( 각 포인트의 geometric feature들과 이에 상응하는 image feature pixels)은 concatenate되고 symmetric feduction function을 사용해 fixed-size global feature vector를 만들기 위해 다른 network로 보내진다.
우리는 estimation에 single global feature를 사용하는 것을 자제했지만, 여기서는 global densely-fused feature로 각각의 dense pixel-feature를 질 높게 만들어서 global context를 제공한다.

우리는 픽셀 당 결과 feature들을 마지막 network로 보내어서 objs의 6D-OPE를 예측한다.
다시말해, 우리는 각각의 densely-fused feature로부터 하나의 pose를 예측하기 위해 이 네트워크를 train시킬 거다.
결과는 feature당 하나씩 pose를 예측한 P의 set이 된다.
이는 우리의 첫번째 학습목적을 정의한다. Section 3.5에서 이에 대해 설명한다.
우리는 이제 [41]에서 영감을 얻어서 self-supervised 방법으로 best prediction을 선택하는 것에 대해 설명한다.

### _Per-pixel self-supervised confidence_

우리는 특정한 context를 바탕으로 가장 좋은 가설이 될지 모르는 pose estimation을 결정하는 우리의 pose estimation network를 훈련할거다.
이를 위해, 우리는 pose estimation predictions와 같은 각각의 prediction에 대한 confidence score ci 를 출력하기 위해 네트워크를 수정한다.
우리는 다음 섹션의 마지막에 보게 될 것처럼 이 두 번째 학습 목표를 전반적인 학습 목표에 반영해야 할 것이다.

<br/>

## 3.5 _6D Object Pose Estimation_

전체의 네트워크 구조를 정의하기 위해, 우리는 이제 학습 목표를 자세히 살펴본다.
우리는 pose estimation loss를 ground truth pose에서 objs model에서 샘플된 point들과 예측된 pose로 인해 transform된 같은 모델에서 일치하는 point들 간의 distance로 설정한다.
구체적으로 말하자면, dense-pixel 당 예측을 최소화하기 위한 loss는 아래 식(1)과 같이 정의된다.

$$
\begin{align*}
L_i^p &= \frac{1}{M}\sum_j ||px_j-\hat{p}_ix_j|| \\
&=\frac{1}{M}\sum_j || (Rx_j+t)-(\hat{R}_ix_j+t_i) ||
\end{align*}
$$

* $x_j$ : obj의 3D 모델로부터 랜덤하게 3D point들로 선택된 M의 j번째 point
* $p$ : grount truth pose $p=\[ R \| t \]$ 
* $\hat{p}$ : $\hat{p} = \[ hat_R \| hat_t \]$로 i번째 dense-pixel의 fused embedding으로 부터 만들어진 예측된 pose

**요점을 다시 상기시켜주기!**

**_카메라 frame을 기준으로 object의 pose(frame)의 Transformation을 구한게 p다.
아직 코드를 다 안봐서 애매하지만, 그림 1에서 i번째 dense-pixel(얘가 pixel-wise dense fusion인지, 
pixel-wise feature인지 애매해 근데 아마 loss를 구하는 부분이니까 후자일듯?)에서 embedding들이 융합된 것들을 찾을 수 있는데, 
근데 이것들이 각각 $x_1,y_1$ , $x_2,y_2$ 이런걸 같는단말야 이 각각의 것들이 각각의 $p_i$를 갖는건가?? 잘 모르겠다**_

위 loss function은 오직 비대칭의(asymmetric) objs에 대해 잘 정의된다. obj shape and or texture가 단 하나의 canonical frame을 결정할 때를 얘기한다.
Symmetric objs는 하나 이상 또는 어쩌면 무한개의 cannonical frame을 가진다. 이는 학습 목표를 모호하게 한다.
그러므로 symmetric objects에 대해서, 우리는 대신에 예측된 model orientation에서의 각 포인트들과 ground truth model의 closet(근접한) 포인트 사이에서의 distance를 최소화 시킨다. 아래 식처럼 말이다.
{비대칭은 뭔가 모양이나 그런게 진짜 비대칭인 걸 이야기 하는건가..? 대칭은 뭔가 데칼코마니처럼 대칭인걸 이야기하는 건가..?}

$$
L_i^p=\frac{1}{M}\sum_j \min_{0<k<M} || (Rx_j+t) - (\hat{R}_i x_k + \hat{t}_i) ||
$$

dense-pixel 당 모든 예측된 pose들을 아우르는 최적화는 아래 식처럼 dense-pixel 당 loss들의 합을 최소화 하는 것이다.

$$
L=\frac{1}{N}\sum_i^N L_i^p
$$

위 식은 asymmetric obj와 symmetric obj의 Lip를 각각 다 구한다음에, 다 더해주는 듯 하다. 이때 N은 입력으로 가져갈 Pointcloud의 수를 이야기하는 것 같다.
그렇게 계산된 Lip 의 평균을 취하는거임. 근데 여기에 confidence를 weight로 더 주어서 아래 식을 만듦.

$$
L=\frac{1}{N}\sum_i^N (L_i^pc_i-w\log(c_i))
$$

※ N은 segment의 P element들로부터 랜덤하게 샘플된 dense-pixel feature의 수를 이야기한다.

그러나, 앞서 설명한 것과 같이 우리는 우리의 네트워크가 dense-pixel 당 예측들 사이에서 confidence를 조율하는 것을 배우길 바란다.
그러기 위해, 우리는 dense-pixel 당 loss에 dense-pixel confidence의 가중치를 주었고, second confidence reqularization term을 추가한 것이다.
말이 어려운데, 그냥 코드에서 보다시피 입력으로 가져갈 pointcloud의 수를 이야기한다.
$w$는 balancing hyper parameter이다.
$c_i$ 는 confidence 이다.
직관적으로 low confidence는 low pose estimation loss를 결과로 도출하지만, second term으로부터 높은 panalty를 부과하게 된다.

이게 무슨말이냐?? 예를들어 $c_i$ 가 1과 100이 있다고 해보자.
그러면 $\log(c_i)$는 $c_i$ 가 1일 때 0, 100일 때 2가 된다. $w>0$이라면, 뒤에 second term 
즉, $-w\ast\log(c_i)$는 $c_i$가 100일때 보다 1일 때 더 큰 값이 되어서 최종적인 Loss가 $c_i$가 1일 때 더 크게 된다는 이야기다.

우리는 $c_i$ 가 큰, high confidence를 가진 pose estimation을 final output으로써 사용한다.

<br/>

## 3.6 _Iterative Refinement_

ICP [2] 는 많은 6D-OPE estimation 방법들에 사용되는 강력한 refinement 방법이다.
그러나, ICP는 종종 real-time에 대해선 충분히 효과적이지 않다.
그래서 우리는 final pose estimation 결과를 빠르고 robust한 방법으로 향상시킬 수 있는 iterative refinement module을 바탕으로 한 neural network를 제안한다.

목표는 다음과 같다.
"네트워크가 iterative 방법으로 자신의 pose estimation을 바로잡을 수 있게 하는 것"
여기서 challenge는 새로운 예측을 하는 것과 반대로 이전의 예측을 개선하기 위해 네트워크를 훈련시키는 것이다.
이를 하기 위해서, 우리는 이전의 iteration에서 이루어진 예측을 다음 iteration의 input의 한 파트로 포함해야한다.
우리의 핵심 아이디어는 이전에 예측된 $Pose = \[ R \| t \]$를 target obj의 canonical frame(기준 프레임)의 예측으로써 생각하는 것이고, 
input point cloud를 이 예측된 canonical frame ( R,t )으로 transform하는 것이다. -> 이부분이 코드에서 new_targets, new_points를 담당하는 듯.

아마 estimator만 쓰면 camera2obj의 rotation, translation이 아주 만족스럽진 않을 것임. 
다시 말해서, 내 카메라 frame을 기준으로 카메라 frame에 Rotation, translation을 취한 predicted target obj frame과 target obj의 frame과 아마 일치하지 않을 것임.
이를 개선하기 위해서 refiner를 사용하는 것임.
자 근데, 여기서 알아야할 것이 있다.
predicted target obj frame과 target obj의 frame은 내 카메라를 기준으로 서술되어지고 있다.
그러나 refiner에서는 이 시점을 완전히 돌려서 카메라 시점이 아닌 predicted target obj frame과 target obj의 frame을 각각 일개의 Body로 보는거지.
왜 이렇게 생각하냐면, estimator의 loss 코드에서

```python
new_points = torch.bmm((points - ori_t), ori_base).contiguous()
new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()
```

이 부분을 생각해보면 points는 camera에서 서술, ori_t는 camera에서 서술, ori_base는 $R_{camera2obj}$ 로 서술된다.
그러면 코드에서 **torch.bmm((points - ori_t), ori_base)** 은 다음과 같이 쓰인다.

![식1](\assets\images\다양한 공부\논문\컴퓨터 비전\DenseFusion\식1.png)

**torch.bmm((new_target - ori_t), ori_base)**은 다음과 같이 쓰인다. 

![식2](\assets\images\다양한 공부\논문\컴퓨터 비전\DenseFusion\식2.png)

object (mesh) frame에서의 시점으로 변경될 것이다.
그러나 실제로 new_target 이라는 frame은 obj와 일치하지 않고 일개의 {target1} 이라는 frame이라고 봐보자.
그리고 new_points 또한 new_target과 같은 연산으로 정해졌으니 {target1}이라는 frame으로 본다고 해보자.
refiner 네트워크는 new_points, emb,idx 의 입력을 받고 pointcloud인 new_points로 geometry embedding을 뽑고 emb로 color embdding을 뽑는다. 
그리고 전역 평균 풀링을 이용하고 어쩌구 저쩌구 해서 pred_r (쿼터니언) pred_t (translation residual)을 얻어낸다.

loss_refiner에서 unit quaternion을 구하고 Rotation matrix를 구하는데 Rotation matrix는 {another}frame에서 {obj(mesh)} frame으로의
$R_{target1 2 obj}$라고 칭하면 코드에서 tranpose로 인해 $R_{obj2target1}$가 된다. 

```python
pred = torch.add(torch.bmm(model_points, base), pred_t)
```

위 코드는 다음 식을 뜻한다.

![식3](\assets\images\다양한 공부\논문\컴퓨터 비전\DenseFusion\식3.png)

이렇게 new_target_{target1} 과 $pred_{target1}$의 distance를 구하고 return한다.

이 때, new_target과 new_points를 다시 구하게 되는데, 코드는 위와 다를 바 없다.

```python
new_points = torch.bmm((points - ori_t), ori_base).contiguous()
new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()
```

한번 더 이를 해석하면 반복일 뿐이니 해석해보자.

**torch.bmm((points - ori_t), ori_base)** 은 (points_{target1} - ori\_t_{target1}) \ast R_{target1\;2\;obj} = new\_points_{obj} 이렇게 ..
**torch.bmm((new_target - ori_t), ori_base)** 은 (new\_target_{target1} - ori\_t_{target1}) \ast R_{target1 \;2 \;obj} = new\_target+{obj} 이렇게

new_ponts와 new_target을 다시 구할 수 있으며
이때 사용된 $R_{target1 \;2\; obj} = R_{1(1 \;iter)} , t_{target1} = t_{1(1\; iter)}$ 로 생각한다.
또 다시 new_points 또한 new_target과 같은 연산으로 정해졌으니 {target2}이라는 frame으로 본다고 해보자.
그리고 refiner해서 $R_{target2 \;2 \;obj} = R_{2(2 \;iter)} , t_{target2} = t_{2(2 \;iter)}$ 으로 구한다.

이렇게 K iteration을 반복하면 $R_{k(k\; iter)} , t_{k(k \;iter)}$ 까지 구할 수 있게된다.
이때 $R_{k(k \;iter)} , t_{k(k \;iter)}$ 는 각각 rotation residual, translation residual 이라 한다.

iteration을 반복할수록 new_target 과 new_points의 distance의 값이 작아질 것이고
prediction pose $hat_p = \[R_K \| t_K \] \[R_{K-1} \| t_{K-1}\] \ast ... \ast \[R_1 \| t_1\]\[R_0 \| t_0\]$ 이 될 것이다.
이를 네트워크의 loss로 취급하여 역전파를 실행해 loss의 변화도를 측정하고 일정 조건이 될 때마다 optimizer를 실행한다.


이 부분을 생각해보면 points는 camera에서 서술, ori_t는 camera에서 서술,  ori_base는 $R_{camera2obj}$ 로 서술된다. 
왜냐하면 Rotation 행렬이 100% 완벽하게 정확한 예측이 됬다면, $R_{camera2obj} 이겠지만 실제로 그러지 못하기 때문에 another로 둔다.
그러면 torch.bmm((points - ori_t), ori_base) 은 ($points_{camera}$ - ori_t_camera)*$R_{camera2another} = points_{another}$ 이렇게
torch.bmm((new_target - ori_t), ori_base) 은 (new_targetcamera - ori_tcamera)*$R_{camera2another}$ = new_targetanother 이렇게 {another} 이라는 frame에서의 시점으로 변경될 것이다.

refiner 네트워크는 new_points, emb,idx 의 입력을 받고 pointcloud인 new_points로 geometry embedding을 뽑고emb로 color embdding을 뽑는다. 
그리고 전역 평균 풀링을 이용하고 어쩌구 저쩌구 해서 pred_r (쿼터니언) pred_t (translation residual)을 얻어낸다.
그리고 loss_refiner에서 unit quaternion을 구하고 Rotation matrix를 구하는데 이 Rotation matrix는 {another} frame에서 {obj(mesh)} frame으로의 Ranother2obj 라고 칭하면
코드에서 tranpose로 인해 Robj2another가 되고 pred = torch.add(torch.bmm(model_points, base), pred_t)를 살펴보면
torch.add(torch.bmm(model_points, base), pred_t)는 pred = ( model_pointsobj * Robj2another ) + pred_tanother 로 predanother 이 된다. 
이렇게 해서 new_targetanother 과 predanother 의 distance를 구하고 return한다. 이 때, new_target과 new_points를 다시 구한다.

```python
new_points = torch.bmm((points - ori_t), ori_base).contiguous()
new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()
```

한번 더 이를 해석하면 반복일 뿐이니 해석해보자.
그러면 torch.bmm((points - ori_t), ori_base) 은
(pointsanother - ori_tanother)*Ranother2another = pointsanother 이렇게
torch.bmm((new_target - ori_t), ori_base) 은
(new_targetcamera - ori_tcamera)*Rcamera2another = new_targetanother 이렇게
{another} 이라는 frame에서의 시점으로 변경될 것이다.

이를 네트워크의 loss로 취급하여 역전파를 실행해 loss의 변화도를 측정하고
일정 조건이 될 때마다 optimizer를 실행한다.


위에서 변형된 (transformed) input point cloud는 estimated pose를 encode한다.
그리고 우리는 transformed point cloud를 network로 되돌려 보내고 이전에 예측된 pose를 바탕으로 residual pose를 예측한다.
이 과정은 반복적으로 적용될 수 있고, 잠재적으로 각 반복마다 더 나은 pose estimation을 만든다.

이 과정은 그림 2에 나와있다.

구체적으로 우리는 main network의 초기 pose estimation을 감안하여 refinement를 수행하기 위한 pose residual estimator 전용의 network를 훈련시켰다.
각 iteration에서 우리는 main network로부터 나온 image feature embedding을 다시 사용하고 새로운 transformed point cloud를 위해 계산된 geometric features로 dense fusion을 수행했다.
pose residual estimator는 입력으로써 fused pixel feature set의 global feature을 사용한다.
K iteration후에, 우리는 식(5)처럼 iteration 당 예측의 축적을 통해 최종 pose estimation을 얻는다


$$
\hat{p} = [R_K | t_K ] \ast [R_{K-1} |  t_{K-1}] \ast ... \ast [R_0 | t_0]
$$

pose residual estimator은 main network와 공동으로 학습된다.
그러나, 학습의 초반의 pose estimation은 매우 노이지 해서 의미있는 것을 배울 수 없다.
그래서 실제론, 공동 훈련은 main network가 어느정도 괜찮은 pose estimation이 나오는 (수렴) 때 부터
이 pose residual estimator가 같이 학습될 것이다.
