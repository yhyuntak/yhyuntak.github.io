---
title: "pvnet-rendering 기본 설치"
categories:
  - 딥러닝
toc: false

---

# pvnet-rendering을 사용하기 위한 기본적인 설치방법
---

1. pvnet-rendering을 다운받고, blender 2.79a 버전을 다운로드한다.
2. blender 2.79a 버전에서 내장python에 library들을 설치해야한다.

  기본적으로 get-pip.py를 /blender-2.79a-linux-glibc219-x86_64/2.79/python/bin 에 넣고, bin의 위치로 터미널을 들어가서 아래의 명령어를 실행.

  ```python
  ./python3.5m get-pip.py
  ./python3.5m -m pip install easydict
  ./python3.5m -m pip install transforms3d
  pip3 install lmdb
  pip3 install easydict
  pip3 install OpenEXR
  ```

  이때, OpenEXR이 오류가 날 수 있다. OpenEXR을 설치시 기본적인 라이브러리가 필요한 듯하다. 오류가 나면 아래의 명령어로 설치해 주자.
  ```python
  sudo apt-get install libopenexr-dev
  sudo apt-get install openexr
  ```
  
  이렇게 설치를 마무리하면, pvnet-rendering을 사용하기 위한 기본적인 설치가 완료된다.

  * 추가
  
    fuse 코드를 따다보면 from plyfile import PlyData 에서 plyfile이란 패키지가 없는 것을 알 수 있는데, 이는 .py가 아닌 pip로 설치하는 것이다. 
    따라서 다음 명령어를 사용해 따로 설치해주자.

    ```python
    pip3 install plyfile
    ```
    
    (참고로 plyfile은 blender 내장python이 아닌 우분투python 혹은 conda python에 설치할 것)


<br/><br/><br/>

# pvnet이 돌아가는 순서
---

1. fuse.sh 
2. fuse/fuse.py 의 run()
3. prepare_dataset_parallel()
4. prepare_dataset_parallel의 내부에서 collect_linemod_set_info를 실행 {아직 미정이지만 R,T 정보를 저장하는 듯}
5. read_txt_and_extract_image_name를 통해서 train_image_list를 뽑아냄. dictionary 완성 그리고 pkl 저장 
6. randomly_read_background 실행 
7. background dir내의 있는 이미지 이름들을 다 가져와서 list화 함. 그리고 pkl로 저장
8. 생성하고자하는 수만큼 iteration을 돌리는데 여기서 prepare_dataset_single가 쓰임. prepare_dataset_single 실행 
9. collect_linemod_set_info 를 통해서 database를 가져옴
10. randomly_sample_foreground, randomly_read_background 를 통해 random하게 augmentation 
11. use_regions 를 실행하여 배경이미지에 랜덤하게 object를 배치하고 마스크도 배치해줌. 그리고 bbox의 중심점 또한 표현해줌
12. save_fuse_data를 실행하여 이미지와 마스크를 저장하고 pkl파일안에 bbox의 중심점과 pose를 저장한다.
