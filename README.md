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

```python
df = pd.DataFrame(data = boston.data, columns = boston.feature_names) 
df = pd.concat([df, y_target], axis = 1)
df.head()

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
```

<img width="421" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/0229b316-b2dd-4a96-9ebe-bd46b101a064">

<img width="321" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/04d6a125-e3b0-4949-9273-308be3fd9752">

<img width="345" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/07d70633-ae2c-4b83-839d-f7c98c82a1d1">

* CRIM, ZN, B열의 데이터가 일부 값에 심하게 치우쳐 분포되어 있는 경향을 보이고 있음

* MEDV는 정규분포를 가지고 있으며, CHAS(이산 변수)를 제외한 다른 변수들은 정규 분포 또는 Biomodal 분포를 보이고 있음

# EDA(Exploratory Data Analysis)
 # 이상치 확인 및 제거

```python
null_df = df.isnull().sum().to_frame('Null')
null_df
```

 <img width="593" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/d075f279-a6fd-4fa8-a674-ee5762ded5c5">

* CRIM, ZN, CHAS, RM, DIS, PTRATIO, B, LSTAT, MEDV 예측변수에 이상치가 존재하고 있음

* CRIM, ZN, B : 해당 변수들의 데이터 값은 이상치가 아닌 데이터 셋에 존재하는 대다수 값으로 보임

# EDA(Exploratory Data Analysis)
 # 데이터 상관관계 분석

```python
df.corr()
```

 <img width="577" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/fda8fd70-a3a8-4712-8ed2-f809a0b6f369">

* 상관관계 Matrix에서는 LSTAT, RM이 매우 높은 연관성을 보이고 있음

* INDUS, TAX, CRIM, NOX, PTRATIO 열도 예측 변수로 사용하기 좋은 성능 척도인 0.5 이상의 상관 점수를 보이고 있음

* 모델 생성시 종속 변수와 상관 점수가 높은 LSTAT, RM, INDUS, TAX, NOX, PTRATIO 예측 변수들을 모델 구축에 활용하였음

# PREPROCESSING
 ## 모델 학습 전 데이터 전처리

```python
x_data = df[['LSTAT', 'RM', 'INDUS', 'TAX', 'CRIM', 'NOX', 'PTRATIO']]
y_data = df['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3) # 훈련 셋(7): 테스트 셋(3)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

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

 * 라쏘 회귀(Lasso Regression)
   * 라쏘 회귀는 기존 선형 회귀에 추가적인 제약을 주는 방식으로 과적합을 방지함
   * MSE가 최소가 되게 하는 가중치와 편향을 찾으면서 동시에, 가중치들의 절댓값의 합이 최소가 되게함 (즉 모든 원소가 0이 되거나 0에 가깝게 되도록 해야 함)
   * L1-norm


* 릿지 회귀(Ridge Regression)
  * 랏쏘 회귀와 매우 유사하지만 패널티 항에 L1-norm 대신에 L2-norm 패널티를 가짐
  * 라쏘는 가중치들이 0이 되지만, 릿지의 가중치들은 0에 가까워질 뿐 0이 되지는 않음
  * 일부분만 중요하다면 라쏘가, 특성의 중요도가 전체적으로 비슷하다면 릿지가 좀 더 괜찮은 모델임
  * 따라서 해당 데이터 셋에서는 LSTAT, RM이 다른 변수에 비해 상관성이 높게 나왔으므로, 랏쏘 회귀가 더 적합함
  * L2-norm

 <img width="565" alt="image" src="https://github.com/shinho123/Boston-house-price-prediction/assets/105840783/d29158c8-afcb-4d7f-89ec-c72909798bdc">

 * 최적의 하이퍼파라미터 추정 : GridSearchCV
   * max_depth : 10, max_features : 2, min_sample_leaf : 4, min_samples_split : 4, n_estimators : 30

 * Random Forest 회귀 모델은 결정 계수가 0.84로 이전의 회귀 모델과 비교할 때, 성능이 가장 우수하게 나타나고 있음

 * 모델의 변수 중요도를 확인한 결과 LSTAT, RM의 중요도가 가장 높게 나타나고 있음

 * 상위 2개의 변수를 제외한 나머지 변수들의 중요도는 유사하고 나타나고 있음

# CONCLUSION

* 목적
  * Boston house prices은 보스턴의 교외 및 마을의 정보를 설명
  * 주어진 독립 변수들을 다양하게 조합해보고 머신러닝의 대표적 지도학습인 회귀(Regression)분석 사용하여 효과적인 예측을 수행하고자 함
  * 다중 선형회귀분석, Lasso 회귀분석, Ridge 회귀분석, Random Forest Regressor 모델을 사용해보고 예측 성능을 평가하여 효율적인 회귀모델을 구축 

* 결론
  * 모델 비교 및 평가
    * 다중 선형, Lasso, Ridge, 랜덤포레스트 회귀 모두 결정 계수가 0.7이상으로 예측 변수에 대한 종속 변수의 결과를 잘 예측함.
    * 다중 선형, Lasso, Ridge 모델의 경우 결정 계수(조정된 결정 계수) 값은 0.7로 비교적 비슷했으나, Lasso 모델이 다른 두 모델과 비교하여 과적합을 많이 줄여 안전성이 가장 높은 모델(오차가 가장 적음)로 파악됨
    * 앞서 비교한 3가지 모델과 랜덤 포레스트 회귀 모델의 비교 결과 결정 계수가 0.84로 랜덤 포레스트 회귀 모델이 성능이 가장 우수한 모델로 파악됨
   
  * 모델 결과 해석
    * 랜덤 포레스트 모델을 통해 변수 중요도를 파악한 결과 주택 가격에 가장 영향을 많이 주는 변수는 LSTAT(하위계층 비율), RM(방의 개수)로 EDA에서도 LSTAT, RM 변수가 종속 변수와 가장 많은 상관성을 보여줌
    * 그 다음으로 CRIM(범죄율), PTRATIO(학생/교사 비율), TAX(재산세율), INDUS(토지비율), NOX(일산화질소 지수)순으로 나타나고 있으며, 결국 주택가격은 주택 인근의 사회현상(범죄율)이나 인프라(학생/교사 비율)의 영향도 복합적으로 작용하는 것을 의미함
