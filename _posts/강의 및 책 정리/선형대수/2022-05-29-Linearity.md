---
title: "Linearity"
categories:
  - 선형대수
toc: true
---
  
# Linearity

## 정의

함수 $f(x)$를 정의하자. Linearity를 만족하기 위해선 만족해야하는 성질이 2가지가 있다.  

1. **Superposition** : $f(x_1+x_2)\,=\,f(x_1)+f(x_2)$
2. **Homogeniety** : $f(ax)=af(x)$

위 두 성질을 한번에 묶는다면 다음과 같다.

$$
\begin{aligned}
f(a_1 x_1+a_2 x_2) = a_1 f(x_1) + a_2 f(x_2)
\end{aligned}$$

위 최종 식을 만족해야만 Linearity를 갖는다고 이야기할 수 있다. 

## 예시
$f(x)=y=mx$가 있다고 하자. 그리고 최종 식을 적용해보자.

$$
\begin{aligned}
f(a_1 x_1 + a_2 x_2) 
&= m(a_1 x_1 + a_2 x_2) \\
&= a_1 m x_2 + a_2 m x_2 \\
&= a_1 f(x_1) + a_2 f(x_2)
\end{aligned}
$$
따라서 $f(x)=y=mx$는 Linearity를 갖는다.  

여기서 핵심은 **원점을 지나는 직선만이 선형성을 갖는다는 것이다.**

