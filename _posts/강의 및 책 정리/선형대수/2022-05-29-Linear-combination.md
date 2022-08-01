---
title: "Linear combination"
categories:
  - 선형대수
toc: true
---
  
# Linear combination

<br/>

## 정의

다음의 두개의 3차원 column vector $v,w$가 있다고 하자.

$$
v=\begin{bmatrix}
a_1\\
b_1\\
c_1
\end{bmatrix}, 
w=\begin{bmatrix}
a_2\\
b_2\\
c_2
\end{bmatrix}
$$

두 벡터에 상수 $\alpha,\;\beta$를 곱한 후 더하자. $\alpha,\;\beta$에 contraints가 없다면, 이것을 Linear combination이라고 부른다. 

<br/>

### 제약이 있으면?
제약에 따라 combination의 이름이 바뀐다.
1. [Affine combination](https://en.wikipedia.org/wiki/Affine_combination) : $\Sigma a_i = 1$
2. [Convex combination](https://en.wikipedia.org/wiki/Convex_combination) : $\Sigma a_i = 1\quad and \quad a_i >= 0$ 

두 상수를 곱한 후 더하면 다음과 같이 식이 전개된다.

$$\begin{aligned}
\alpha v + \beta w 
&= \alpha \begin{bmatrix}
a_1\\
b_1\\
c_1
\end{bmatrix} + \beta \begin{bmatrix}
a_2\\
b_2\\
c_2
\end{bmatrix}
&= \begin{bmatrix}
\alpha a_1 + \beta a_2\\
\alpha b_1 + \beta b_2\\
\alpha c_1 + \beta c_2\\
\end{bmatrix}
\end{aligned}
$$

<br/>

## 꼭 알아야 할 것

[위](#정의)의 형태는 아주 간단하게 벡터의 곱 형태로 표현가능하다. 이건 다들 아는 사실이다.

$$
\begin{bmatrix}
v,w
\end{bmatrix}\begin{bmatrix}
\alpha\\ \beta
\end{bmatrix}
$$

그러나 **정말 중요한 것**은 이제는 벡터의 곱으로 보지말고, 각 column vector에 상수배한 것을 결합한 "Linear combination(선형 결합)"을 떠올리자는 것이다.
이것은 공부를 해나갈수록 굉장히 중요하고도 핵심적인 개념이 된다.

특히 고차원으로 갈 수록 연립방정식을 풀 때 기하학적으로 문제를 풀어낼 수 없다. 당장 4차원만 하더라도 
우리 머릿속에선 그림이 그려지지 않기 때문이다. 이같은 경우는 우리가 행렬을 row vector들의 선형 결합으로 바라볼 때 발생한다. 그러나 이 문제를 **column vector들의 선형결합**으로
바라본다면, 문제가 아주 쉬워지게 된다. 

따라서 우리는 이제부터 **행렬간의 곱을 column vector의 선형 결합으로 바라보는 연습**을 해야한다. 

