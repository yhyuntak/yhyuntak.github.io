class Person:
  def greeting(self):
    print('안녕하세요.')


class Student(Person):
  def greeting(self):
    print('안녕하세요. 저는 파이썬 코딩 도장 학생입니다.')


class Student2(Person):
  def greeting2(self):
    print('안녕하세요. 나는 무관해요.')


james = Student()
james.greeting()
james2 = Student2()
james2.greeting2()