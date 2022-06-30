---
title: "Correlation Matrix"
categories:
  - 선형대수
toc: true
---
  
# Correlation Matrix

## 정의

다음의 벡터 $A$가 존재한다고 생각하자. 

$$
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
$$

그리고 벡터 $A$를 $A^T A$형태로 outer product를 진행하면 행렬 $R$을 얻을 수 있다.

$$
\begin{aligned}
R &= A^T A \\
&= \begin{bmatrix} x_1^T x_1 & x_1^T x_2 & \dots & x_1^T x_n \\
x_2^T x_1 & x_2^T x_2 & \dots & x_2^T x_n \\
\vdots & \vdots & \dots & \vdots \\
x_n^T x_1 & x_n^T x_2 & \dots & x_n^T x_n \\
\end{bmatrix} 
\end{aligned}
$$

이 행렬 $R$을 Correlation Matrix 라고 부른다. 여기서 각 요소들 $x_i^T x_j$은 $x_j$가 $x_i$방향으로 얼만큼의 성분을 갖는지를 표현한다고 생각하면 된다.

그래서 만약 성분을 갖지 않으면, 직각이 되므로 $x_i^T x_j = 0$이 되어버린다. (여기서 개인적으로 궁금한건 이렇게 0이 되어버리면 두 벡터들끼리는 연관이 없음을 시사하는 것일까?)

