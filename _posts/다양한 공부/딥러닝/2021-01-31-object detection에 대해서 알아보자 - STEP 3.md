---
title: "object detection에 대해서 알아보자 - STEP 3"
categories:
  - 딥러닝
  - object detection
toc: True

---

[이전 글](https://yhyuntak.github.io/%EB%94%A5%EB%9F%AC%EB%8B%9D/object%20segmentation/object-segmentation%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-STEP-2/)에 이어서 계속해서 리뷰를 하겠습니다.

우리는 STEP 2에서 SS를 이용하여 object가 이미지에 어디에 있는지 제안하는 것까지 알아봤습니다.

|![그림 1](/assets/images/다양한 공부/딥러닝/object detection/step3/그림 1.gif)|
|:--:|
|_그림 1_|

그림 1을 보면, 영역들을 제안하는 것을 볼 수 있지만, 영역들안에 object가 확실하게 있는지 없는지에 대해선 모릅니다!

이 프로젝트의 궁극적인 목표는 object detection입니다!
SS를 이용한 object detection의 기본적인 순서는 STEP 1과 같습니다. 
단, STEP 1은 윈도우 슬라이딩을 통해서 비효율적으로 ROI를 추출했지만, 이제는 SS를 통해서 "object가 있을 법한 영역"을 제안하는 것이죠.
SS로 제안된 영역들에 대해서만 분류를 진행하고 bbox,label,score를 얻어서 STEP 1에서 한 것처럼 non-maximum suppression을 통해 결과를 그림 2처럼 얻어내면 끝나는 겁니다!

|![그림 2](/assets/images/다양한 공부/딥러닝/object detection/step3/그림 2.png)|
|:--:|
|_그림 2_|

그림 2의 왼쪽이 SS를 통해 제안 받은 영역들이라면, 오른쪽은 분류기를 지나 non-maximum suppression을 거쳐서 나온 최종적인 bbox라고 생각하면 됩니다.
