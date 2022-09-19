---
title: "파이썬 if __name__ == '__main__'  은 왜 쓸까?"
categories:
  - python
toc: true
---

파이썬을 사용하다보면 **_if __name__ == '__main__'_**: 으로 시작하는 부분을 자주 만나는데, 
이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해서 사용한다고 한다.

exam.py 에 아래의 코드를 저장해보자.

```python
print('모듈 시작')
print('exam.py __name__:', __name__)    # __name__ 변수 출력
print('모듈 끝')
```

같은 폴더 내에 exam2.py를 만들고 아래의 코드를 저장하자.

```python
import exam
print('exam2.py __name__:', __name__)    # __name__ 변수 출력
```

그리고 python exam2.py를 하면, 다음과 같은 결과를 얻는다.

![그림1](/assets/images/프로그래밍 언어/파이썬/exam그림1.png)

import를 통해 모듈을 가져오면, 그 모듈을 무조건 한번 쫙 읽는 것 같다..
그래서 exam.py에 있는 코딩이 한번 실행된 것이고 exam2.py에 있는 print가 작동된 것이다.

이게 왜 이렇게 될까?? __name__은 현재 모듈의 이름을 읽어오는데, import exam을 통해서 읽어온 exam.py의 모듈은 exam이므로 그림 1과 같은 결과가 뜬 것이다.
그리고 exam2.py를 실행하였으므로 __name__은 자기 자신은 exam2 이렇게 가지는게 아니라 '__main__' 으로 갖는다. 

그럼 수없이 많은 예제들에서는 import를 통해 많은 모듈들을 읽어왔었고 그때마다 코드들을 한번 쭉 훑고 분명 print()가 하나라도 있었을텐데 왜 한번도 뜨지 않은걸까?? 

그 이유는 바로 if __name__ == '__main__' : 에 있다. 

exam.py에 코드를 아래와 같이 수정하자.

```python
def main() :
    print('모듈 시작')
    print('exam.py __name__:', __name__)    # __name__ 변수 출력
    print('모듈 끝')

if __name__ == '__main__' :
    main()

```

그리고 exam2.py를 아래와 같이 수정하자

```python
import exam
def main():
    print('exam2.py __name__:', __name__)    # __name__ 변수 출력
if __name__ == '__main__':
    main()
```

그리고 python exam2.py를 해보면 그림 2와 같은 결과가 나온다.

![그림2](/assets/images/프로그래밍 언어/파이썬/exam그림2.png)

왜 exam.py에도 print()가 있는데 실행이 안되었을까?? 그 이유는 import exam을 통해서 exam.py를 한번 훑을 때, __name__에는 exam이라는 값이 들어가 있기 때문이다.
즉, exam.py에서  if __name__ == '__main__' 이란 조건문에 부합하지 않은 것이다. 그래서 exam.py의 main()이 실행되지 않았고 print()가 작동하지 않은 것이다.
exam2.py에선 __name__이 당연히 '__main__'이므로 조건문에 부합되어 main()이 실행된 것이다

이런 이유로 인해 if __name__ == '__main__'  조건문을 써서 **모듈과 내가 정말 실행할 파일을 구분짓고 프로그램의 시작점을 나타내는 것** 같다.


참고 : [https://dojang.io/mod/page/view.php?id=2448](https://dojang.io/mod/page/view.php?id=2448)