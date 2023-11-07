# Boston-house-price-prediction
보스턴 집값 예측 문제 

# Boston-house prices란?
  * Boston house prices은 보스턴의 교외 및 마을의 정보를 설명하고 있음
  * Dataset은 1970년 Boston standard metropolitan statistical area(SMSA)에서 가져옴
  * 총 14개의 속성(Attribute)으로 구성(설명 변수 : 13개, 종속 변수 : 1개)
  * 속성별 데이터 개수 : 506개

# Boston-house prices attributes 설명

<img width="269" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/671173cc-b674-48ad-ba1b-7cb2f95e9a83">

# EDA(Exploratory Data Analysis)
  # 데이터 정보 확인
  
<img width="345" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/07d70633-ae2c-4b83-839d-f7c98c82a1d1">

<img width="321" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/04d6a125-e3b0-4949-9273-308be3fd9752">

* CRIM, ZN, B열의 데이터가 일부 값에 심하게 치우쳐 분포되어 있는 경향을 보이고 있음

* MEDV는 정규분포를 가지고 있으며, CHAS(이산 변수)를 제외한 다른 변수들은 정규 분포 또는 Biomodal 분포를 보이고 있음

# EDA(Exploratory Data Analysis)
 # 이상치 확인 및 제거

 <img width="593" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/d075f279-a6fd-4fa8-a674-ee5762ded5c5">

* CRIM, ZN, CHAS, RM, DIS, PTRATIO, B, LSTAT, MEDV 예측변수에 이상치가 존재하고 있음

* CRIM, ZN, B : 해당 변수들의 데이터 값은 이상치가 아닌 데이터 셋에 존재하는 대다수 값으로 보임

# EDA(Exploratory Data Analysis)
 # 데이터 상관관계 분석

 <img width="577" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/fda8fd70-a3a8-4712-8ed2-f809a0b6f369">

* 상관관계 Matrix에서는 LSTAT, RM이 매우 높은 연관성을 보이고 있음

* INDUS, TAX, CRIM, NOX, PTRATIO 열도 예측 변수로 사용하기 좋은 성능 척도인 0.5 이상의 상관 점수를 보이고 있음

* 모델 생성시 종속 변수와 상관 점수가 높은 LSTAT, RM, INDUS, TAX, NOX, PTRATIO 예측 변수들을 모델 구축에 활용하였음

# PREPROCESSING
 ## 모델 학습 전 데이터 전처리

* 예측 변수 선택 : 'LSTAT', 'RM', 'INDUS', 'TAX', 'CRIM', 'NOX', 'PTRATIO'
 1. Train / test set split - 7:3
 2. 예측 변수에 대한 데이터 정규화 사용 : MinMaxScaler
 3. 모델 학습 : Multiple Linear Regression, Lasso Regression, Ridge Regression, Random Forest Regressor
 4. 모델 성능 평가 및 비교

# PREPROCESSING
 ## 모델 평가 지표

 <img width="150" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/ce65ecd5-6ab4-4616-8dd7-fc8b0c944c34">

* Regression 모델 평가 지표 : R^2, Adjusted R^2, MAE, MSE, RMSE

# REGRESSION MODEL EVALUATION
 ## MULTIPLE LINEAR REGRESSION 성능 순서 : 1 → 2 → 3

 <img width="592" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/4d0cdc39-5f99-40bc-8857-a57d426cb52c">

# REGRESSION MODEL EVALUATION
 ## RANDOM FOREST REGRESSOR

 <img width="565" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/d29158c8-afcb-4d7f-89ec-c72909798bdc">

 * 최적의 하이퍼파라미터 추정 : GridSearchCV
   * max_depth : 10, max_features : 2, min_sample_leaf : 4, min_samples_split : 4, n_estimators : 30

 * Random Forest 회귀 모델은 결정 계수가 0.84로 이전의 회귀 모델과 비교할 때, 성능이 가장 우수하게 나타나고 있음

 * 모델의 변수 중요도를 확인한 결과 LSTAT, RM의 중요도가 가장 높게 나타나고 있음

 * 상위 2개의 변수를 제외한 나머지 변수들의 중요도는 유사하고 나타나고 있음
