이 프로젝트에서 구현된 모델은 다음과 같습니다.
1. GinO+WbyT 
2. GinM+WbyT
3. Centralized XGBoost

## 설치

1. `python3.9.12`와 필요한 라이브러리 및 패키지 설치하세요.
   ```
   pip install -r requirements.txt
   ```

## 디렉토리 및 파일 구조

- `main.py`: 연합학습 실험을 실행하기 위한 실행파일 
- `models/`: Directory containing model implementations:
  - `decision_tree.py`: 기본적인 결정 트리 구현과 기능 함수가 포함된 코드 
  - `utils.py`: 연합학습 환경 조성 및 데이터 처리와 같은 유틸리티 함수들이 포함된 코드
  - `loss_function.pyx`: XGBoost 트리를 구성하는데에 필요한 계산 함수가 포함된 코드
  - `xgboost_WbyT.py`: XGBoost 모델 학습 및 예측을 수행하는 코드
- `setup.py`: cython 설정 파일, `loss_function.pyx` 을 컴파일러하고 필요한 라이브러리를 전달합니다.
- `data/`: 데이터 셋이 저장된 디렉토리
- `results/`: 현재 실험 설정과 실험 결과를 각 시점의 앙상블 트리에서의 테스트 정확도와 트리의 노드 수를 텍스트 파일로 저장하는 디렉토리

Arguments:

## 실험 환경 조성 설정변수
- `--seed`: 랜덤 시드 설정 
  
- `--filename`: 데이터 셋(가정: C개의 클래스 값 범위를 [0, C-1]로, 헤더가 존재해야 함) 
  - Options: 'pendigits', 'drybean', 'nursery', 'satellite', 'letter'

- `--joint`: 중앙집중식 학습 또는 연합 학습 설정 
  - Default: True
  - Options:
    - False: 중앙집중식 환경 학습
      - `--method`에서 `GinO`를 같이 설정해야 함.
    - True: 연합학습 환경 학습

- `--n_clinets`: 연합학습 참여 클라이언트 수 
  - Default: 10
  
- `--env`: 클라이언트 데이터 환경 설정 
  - Options: 'IID', 'label_nonIID'

- `--alpha`: Non-IID 데이터 환경에서 불균형 조절 매개 변수
  - `--env`에서 `--label_nonIID`을 설정할 경우 적용됨.
  - 데이터 셋별 alpha 값:
    - pendigits: 1
    - drybean: 1
    - statlog: 1
    - nursery: 1
    - letter: 2

## XGBoost 학습 설정변수
- `--trees`: XGBoost 트리의 수 
  - Default: 100
  
- `--depth`: 트리의 최대 깊이. 
  - Default: 10

- `--count`: 이진 분할에 필요한 최소 데이터 수. 
  - Default: 2

- `--fraction`: 트리를 생성할 때 사용하는 데이터 샘플의 비율
  - Default: 1.0

- `--lr`: 학습률(learning rate)
  - Default: 0.3
  
- `--hes`: Vector-valued XGBoost에서 헤시안 행렬 계산 방법 선택
  - Default: True
  - Options:
    - False: Full Hessian matrix (미구현)
    - True: Diagonal Hessian matrix
    
- `--lamb`: 규제 매개변수
  - Default: 1.0

## 연합 XGBoost 방식 설정변수

- `--method`: 연합학습 환경에서 트리 학습 방법을 선택
  - Default: 'GinM'
  - Options:
    - GinO: 한 클라이언트에서 트리 구조를 생성.
    - GinM: 여러 클라이언트를 순회하면서 크리 구조를 생성.

- `--levelUp`: 클라이언트 당 트리의 추가 높이
  - `--method`에서 `--GinM`을 설정할 경우 적용됨.
    - Default: '1'
    
- `--init_depth`: 첫 번째 클라이언트에서 생성할 트리의 높이
  - `--method`에서 `--GinM`을 설정할 경우 적용됨.
    - Default: '1'
    
## 사용법
먼저, `loss_function.pyx`의 컴파일러를 수행하기 위해 터미널에서 명령어를 실행:

```
python setup.py build_ext --inplace
```

다음 명령어를 통해 코드를 실행:

```
python main.py
```
