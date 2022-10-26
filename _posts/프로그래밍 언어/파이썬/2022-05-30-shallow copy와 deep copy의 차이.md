---
title: "shallow copy와 deep copy의 차이"
categories:
  - python
toc: true
---

최근 [밑바닥부터 시작하는 딥러닝 2](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791162241745&orderClick=LEa&Kc=)를 공부하고있는데, 
코딩속에서 갑자기 self.grads[0][...]=dW 라는 문장이 나오면서 shallow copy와 deep copy에 대한 이야기를 하더군요. 
난생 처음보는 문법이라 이게 뭔가 했는데, 알고 보니 데이터를 할당하는 방식을 다르게 하는 것이었고, 그것이 shallow copy와 deep copy라는 개념으로 파이썬에선 나눠지는 듯 합니다. (제가 잘못 이해한 것일 수 있습니다.)

이 두 copy의 차이점은 무엇일까요?

# Shallow copy
---

shallow copy는 다음과 같이 간단하게 사용할 수 있습니다.

```python
a = [1,2,3]
b = a
```

위 코딩에서 우리는 a라는 변수의 메모리에 [1,2,3] 을 저장했습니다. 
여기서 shallow copy는 b라는 변수가 새로운 메모리주소를 갖고 a라는 변수의 값을 가져와서 저장하는 것이 아니라, 
단순하게 **a라는 메모리 주소를 가르키는 역할**을 한다. 그래서 실제로 주소를 확인하는 id()를 사용해보면,

```python
a= [1,2,3]
b = a
print(id(a))
print(id(b))
```

다음의 결과를 보게됩니다.

```python
139759473549184
139759473549184
```

139759473549184으로 주소가 같은 것을 알 수 있습니다. 
만약 여기서 b의 값을 수정하면, a도 같이 바뀌게 됩니다. 즉, b=a는 **실제 데이터를 복사하지 않는다는 것**을 알아야합니다.

리스트로 선언하고 값을 바꿔도 여전히 같은 메모리를 가리키기 때문에, 값이 똑같은걸 다음과 같이 볼 수 있죠.

```python

a= [1,2,3]
b = a
b[0]=99
print(a)
print(b)
```

* 결과
```python
[99, 2, 3]
[99, 2, 3]
```

더 재밌는건 a를 다른 리스트로 재할당을 하면 b도 따라갈줄 알았는데, 그게 아니라는 점입니다.

```python
a= [1,2,3]
b = a
print("원래 주소")
print(a)
print(b)
print(id(a))
print(id(b))


print("a를 재할당한 결과")
a = [2,4,1]
print(a)
print(b)
print(id(a))
print(id(b))
```

* 결과
```python
원래 주소
[1, 2, 3]
[1, 2, 3]
140008847392384
140008847392384
a를 재할당한 결과
[2, 4, 1]
[1, 2, 3]
140008847392000
140008847392384
```

기존의 a의 주소값을 b가 가져가버렸다는 것이 좀 웃기네요 ㅎㅎ 

<br/><br/><br/>

# Deep copy
---

그럼 Deep copy는 어떨까요? copy 라이브러리의 deepcopy를 이용하면 쉽게 deepcopy를 할 수 있습니다. 먼저 결과를 봅시다.

```python
import copy

a = [1,2,3]
b = copy.deepcopy(a)

print(a)
print(b)
print(id(a))
print(id(b))
```

* 결과
```python
[1, 2, 3]
[1, 2, 3]
140257731666112
140257731648704
```

a,b는 같은 값을 갖지만 주소가 전혀 다른 것을 볼 수 있습니다. 
즉, deepcopy를 이용하면 **같은 값을 갖되, b가 독자적인 메모리 할당을 가져가는 것**을 알 수 있습니다. 당연히 b값을 바꾸더라도 a는 영향이 없습니다.