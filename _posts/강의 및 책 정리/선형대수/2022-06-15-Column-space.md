---
title: "Column space"
categories:
  - 선형대수
toc: true
---
  
# Column space

## 정의
행렬 $A$를 다음과 같이 정의하자.

$$
A = \begin{bmatrix}a_1&a_2&\dots&a_n\end{bmatrix}
$$

Column space란, 행렬 $A$의 column vector들의 모든 linear combination을 의미한다.

$$
\Sigma_{i=1}^{N} c_i a_i \in C(A)
$$

## 알아야할 것
1. $Ax=b$의 방정식이 있을 때 $b$가 $C(A)$에 속하지 않으면, 해 $x$는 존재하지 않는다.
### 예시

$$
\begin{aligned}
\begin{bmatrix}1&0\\5&4\\2&4\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix} 
&= \begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}
&= u\begin{bmatrix}1\\5\\2\end{bmatrix}+v\begin{bmatrix}0\\4\\4\end{bmatrix}
\end{aligned}
$$

위와 같은 식에서 해 $b=\begin{bmatrix}b_1&b_2&b_3\end{bmatrix}^T$가 존재하려면, 원점과 
두 column vector들로 이루어진 평면에 $b$가 존재해야 한다. 
즉, 두 column vector들의 linear combination이 해가 된다는 뜻이다.

여기서 반드시 중요한 것은 **Column space의 개념은 우리가 익숙한 방정식들의 기하학적 개념을 갖는게 아니라는 것**이다.

2. 반대로 얘기하면, 행렬 $A$의 역행렬이 존재해 해가 $x=A^{-1}b$로 존재할 경우, $b$는 항상 Column space에 존재한다.

