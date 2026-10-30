# 온라인 교육 적응 수준 예측: 분류 모델과 결측 데이터 강건성 분석

[English](README.md) | [한국어](README.ko.md) | [简体中文](README.zh-CN.md)

학생의 온라인 교육 적응 수준을 예측하는 메인 실험과, 결측값 대치 후 데이터 분포 및 모델 성능의 변화를 평가하는 추가 실험으로 구성된 머신러닝 수업 프로젝트입니다. 

**개인 공헌: 추가 실험 설계, 실험 코드 작성, 평가 및 시각화.**

## 1 프로젝트 개요

프로젝트는 크게 결측치가 없는 데이터를 사용하는 **메인 실험**과 인위적으로 결측을 생성하는 **추가 실험**으로 구성됩니다.

### 1.1 메인 실험: 온라인 교육 적응 수준 예측

- **문제정의:** 학생의 배경과 온라인 학습 환경을 바탕으로 온라인 교육 적응 수준을 Low·Moderate·High로 예측할 수 있는지 확인합니다.
- **해결방법:** 1,205개 응답에 대해 EDA를 수행하고, 여러 머신러닝 분류 모델을 교차검증으로 비교했습니다. Optuna로 XGBoost와 CatBoost의 하이퍼파라미터를 튜닝한 뒤, 선택한 모델을 독립 테스트 데이터에서 정확도·macro-F1·balanced accuracy(EDA결과, 클래스별 데이터 수가 불균형해서 소수 클래스도 잘 예측하는지 확인하기 위해서 balanced accuracy를 사용)로 평가했습니다.
- **결과물 활용:** 특성의 분포와 연관성을 탐색하고, 모델별 성능과 튜닝 결과를 비교하며, 완전한 입력으로 적응 수준을 얼마나 정확하게 예측하는지 확인할 수 있습니다. 이 결과는 추가 실험의 베이스라인으로도 사용됩니다.

### 1.2 추가 실험: 결측값 대치에 따른 변화 분석

- **문제정의:** 주어진 데이터는 결측치가 존재하지 않는 완전한 데이터 입니다. 하지만 다수의 실제 환경에서는 결측치 없는 완전한 데이터는 대부분 존재하지 않습니다. 따라서 인위적으로 확률적으로 랜덤하게 결측치를 생성합니다. 그 후, 입력 정보의 누락과 이후 대치가 메인 실험 대비 데이터 분포와 예측 성능을 어떻게 바꾸는지 확인합니다.
- **해결방법:** 전체 13개 특성과 중요도 상위 5개 특성에 각각 무작위 결측을 생성하고, 특성별 DNN으로 대치한 결과를 최빈값 대치와 비교했습니다. 결측률과 마스크를 바꿔 반복하며, 복원된 분포와 후속 분류 성능을 원본 무결측 데이터 결과와 비교했습니다.
- **결과물 활용:** Original·Mode·DNN의 분포를 비교하고, 가린 셀의 복원 정확도를 측정하며, 결측률과 결측 집중도에 따라 무결측 데이터 모델 성능과의 차이가 어떻게 달라지는지 분석할 수 있습니다.

## 2 주요 결과 및 대표 이미지

- **프로젝트 기준 성능:** CV로 선택한 튜닝 XGBoost는 무결측 데이터 테스트 362행에서 **정확도 89.23%, macro-F1 0.8773**을 기록했습니다.
- **추가 실험:** All13의 특성별 5% 결측 주 실험에서 가린 테스트 셀의 복원 정확도는 **DNN 82.17%, 최빈값 69.13%**였습니다.
- **강건성:** All13의 DNN 대치 후 평균 분류 정확도는 결측률 5%, 15%, 30%에서 **85.86% → 78.51% → 69.01%**로 감소했습니다. 각 평균은 고정된 분할에서 마스크 5개를 사용한 결과입니다.
- **불확실성:** 두 시나리오와 세 결측률의 총 6개 조건에서 DNN의 평균 분류 점수가 높았지만, 주 5% 실험의 성능 차이 신뢰구간에는 0이 포함됐습니다.



**1. 무결측 데이터 모델 비교.** 학습 데이터의 5-fold CV 점수로 모델 선택의 기준을 마련했습니다. XGBoost_tuned 의 성능이 가장 높은걸 확인할 수 있습니다.

![무결측 데이터 CV 비교](experiment_results/figures/05_cv_model_comparison.png)

**2. Top5 특성 선정.** 학습 데이터 내부 검증에서 순열 중요도를 계산했습니다. 오차 막대는 순열 8회에 대한 표준편차입니다. 교차검증 macro-F1이 가장 높았던 `XGBoost_tuned`를 기준으로 중요도 상위 5개 특성을 선정했으며, 추가 실험에서 이 특성들의 일부 값을 무작위로 결측 처리했을 때 복원 정확도와 분류 성능에 미치는 영향을 평가했습니다.

![학습 내부 검증 순열 중요도](experiment_results/figures/06_permutation_importance.png)

**3. DNN 학습 진단.** 데이터셋에 존재하는 13개의 특성에 대한 크로스 엔트로피 곡선으로 학습 epoch 수를 정하는 내부 검증 과정을 확인할 수 있습니다. 예시 Figure는 Age의 크로스 엔트로피 이고, 나머지 특성에 대한 크로스 엔트로피 곡선 Figure는 experiment_results/figures 에서 확인할 수 있습니다. 

![DNN 학습 곡선](experiment_results/figures/All13__08_dnn_learning_curve_age.png)

**4. 분포 비교.** Original·Mode·DNN 테스트 분포에 같은 범주 색상과 서로 다른 해치를 적용했습니다. 분포 유사성은 셀 단위 복원 정확도와 함께 해석합니다. 전체적으로 Mode 기반 imputation 보다 DNN 기반 결측치 처리가 원래의 데이터셋 분포와 가까운것을 확인할 수 있습니다. 아래 Figure는 Age 특성에 대한 테스트 분포이며, 다른 특성에 대한 테스트 분포 Figure는 experiment_results/figures 에서 확인할 수 있습니다. 

![원본 최빈값 DNN 분포 비교](experiment_results/figures/All13__09_distribution_age_comparison.png)

**5. 결측률 민감도.** 점은 마스크 5개의 평균, 오차 막대는 표준편차입니다. 점선은 무결측 데이터 테스트 정확도입니다.  DNN imputation 방법이 Mode 기반 imputation 보다 정확도가 높은것을 알 수 있으며, 결측치 비율이 높아질 수록 두 방법 모두 정확도가 낮아지는 추세를 확인할 수 있습니다.

![All13 결측률 민감도](experiment_results/figures/All13__13_rate_sensitivity_accuracy.png)

**6. 동일 결측량 비교.** 두 조건 모두 780개 셀을 가려 전체 특성에 분산된 결측과 Top5에 집중된 결측을 비교했습니다. 오차 막대는 마스크 5개의 표준편차입니다. 13개 특성에 대해 임의의 확률로 결측치를 냈을 때와 모델 성능에 영향력이 높은 5개의 특성에 같은양의 결측치를 냈을 때 모두 DNN의 성능이 Mode 기반 결측치 처리보다 높은것을 알 수 있습니다.

![동일 결측량 비교](experiment_results/figures/14_matched_budget_accuracy.png)

</details>

## 3 개인 기여도

프로젝트 전체는 EDA, 분류 모델 비교, 하이퍼파라미터 튜닝을 포함합니다. 이 중 **추가 실험**을 담당했습니다.

- All13·Top5 결측 마스크 프레임워크, 결측률·마스크 반복 검증, 동일 결측량 비교를 설계하고 코드를 작성했습니다. 근거: [노트북](analysis.ipynb), [마스크별 결과](experiment_results/robustness_all_runs.csv), [동일 결측량 결과](experiment_results/matched_budget_all_runs.csv).
- 학습 데이터에만 적합하고 관측값을 보존하도록 최빈값 대치와 특성별 범주형 DNN 대치기를 구현했습니다. 내부 검증으로 학습량을 결정했습니다. 근거: 노트북의 `CategoricalDNNImputer`, `mode_impute_pair`.
- 가린 셀의 복원과 후속 분류 성능을 평가하고, paired bootstrap 구간 및 입력 프로필 그룹 검증으로 불확실성과 일반화를 살폈습니다. 근거: [복원 점수](experiment_results/All13/imputation_quality.csv), [bootstrap 구간](experiment_results/primary_paired_bootstrap.csv), [그룹 검증](experiment_results/profile_group_test.csv).
- 분포 비교, 학습 곡선, 민감도 그래프와 CSV 저장을 구성해 시각적 해석을 기록된 결과와 연결했습니다.

## 4 데이터 및 전처리

- **데이터:** 1,205행, 범주형 입력 13개, 타깃 `Adaptivity Level`. 클래스별 응답은 Low 480개, Moderate 625개, High 100개입니다.
- **구조 점검:** 원래 결측은 없으며 최초 출현 이후 완전 중복행은 949개입니다. 고유 입력 프로필은 236개이고, 그중 19개에는 서로 다른 타깃이 존재합니다. 동일 응답만으로 동일 응답자임을 판단할 수 없어 행을 유지했습니다.
- **분할:** seed 42의 층화 70:30 행 분할로 학습 843행, 테스트 362행을 사용했습니다. EDA와 중요도 분석은 학습 부분에서 수행했습니다.
- **인코딩:** 공백을 정리하고, CV fold 내부에서 범주형 특성을 원핫 인코딩했습니다. Logistic Regression과 KNN에는 인코딩된 입력의 표준화를 적용했습니다.

## 5 방법론과 기술 선택 이유

### 5.1 무결측 데이터 기준 모델

다수 클래스 Dummy 기준 모델과 Logistic Regression, KNN, Random Forest, XGBoost, CatBoost를 비교했습니다. XGBoost와 CatBoost는 동일한 층화 CV 5개 fold에서 Optuna로 각각 12 trials를 수행했습니다. 각 클래스에 같은 가중치를 주는 macro-F1으로 모델을 선택하고, 정확도와 balanced accuracy도 함께 보고했습니다.

Cramer's V는 범주형 변수의 연관성을, 순열 중요도는 특성을 섞었을 때 검증 macro-F1의 감소를 측정합니다. 후자로 테스트 데이터를 사용하지 않고 **Financial Condition, Age, Class Duration, Gender, Institution Type**을 Top5로 선정했습니다.

### 5.2 추가 실험

1. 타깃을 제외하고 무작위 MCAR 결측 마스크를 생성했습니다. All13은 전체 입력에 대해서, Top5는 고정된 중요도 상위 특성을 대상으로 결측 마스크를 생성합니다. 선택 특성별 5%에서 각각 780개와 300개 셀을 가립니다.
2. 최빈값 기준 대치기를 학습 데이터로 적합했습니다. DNN은 다른 12개 특성으로 각 결측 특성을 예측하며, 원핫 입력과 입력 결측의 최빈값 대치 및 결측 지시자를 사용합니다.
3. PyTorch의 64·32 은닉층, ReLU, dropout 0.1, 크로스 엔트로피, Adam을 사용했습니다. 내부 80:20 분할에서 최대 160 epochs, patience 18로 epoch 수를 선택한 뒤, 관측된 전체 학습 행으로 선택 횟수만큼 다시 학습했습니다.
4. 관측값은 유지하고 가린 원래 값은 평가에만 사용했습니다. 후속 분류기는 무결측 데이터 CV에서 정한 파라미터를 고정해 다시 학습했습니다.
5. 행 분할과 DNN 초기화를 고정하고 결측률 5%·15%·30%, 마스크 seed 42–46으로 반복했습니다. 추가 비교에서는 All13 5%와 Top5 특성당 156개 결측을 사용해 총량을 780개로 맞췄습니다.

## 6 평가 방법과 결과

### 6.1 All13 주 실행

seed 42, 특성별 5% 결측 조건입니다. 모든 조건에서 같은 테스트 362행과 튜닝 XGBoost 설정을 사용했습니다. 복원 정확도는 특성별 가린 셀 수로 가중해 전체 가린 테스트 셀의 정답률을 집계합니다.

| 조건 | 테스트 정확도 | 테스트 macro-F1 |
|---|---:|---:|
| Complete | 89.23% | 0.8773 |
| Mode | 82.04% | 0.7898 |
| DNN | 84.25% | 0.7975 |

복원 정확도는 **DNN 82.17%, 최빈값 69.13%**였습니다. 근거: [분류 결과](experiment_results/primary_classification_comparison.csv), [대치 점수](experiment_results/All13/imputation_quality.csv).

테스트 행을 함께 재표집하는 paired bootstrap 1,000회에서 DNN−Mode 정확도 차이는 **+2.21%p, 95% 구간 −1.10~+5.53%p**였습니다. macro-F1 차이는 **+0.77%p, 구간 −3.80~+5.42%p**로, 모두 0을 포함했습니다. [구간 결과](experiment_results/primary_paired_bootstrap.csv)

### 6.2 결측 반복 실험

아래 정확도는 마스크 5개의 **평균(%) ± 표준편차(%p)**입니다. 분할과 튜닝 XGBoost 설정은 고정되어 있으며, 표준편차는 마스크 변동을 나타냅니다.

| 시나리오 | 선택 특성별 결측률 | 최빈값 정확도 | DNN 정확도 |
|---|---:|---:|---:|
| All13 | 5% | 84.59 ± 2.05 | 85.86 ± 1.81 |
| All13 | 15% | 75.52 ± 0.89 | 78.51 ± 1.83 |
| All13 | 30% | 68.45 ± 1.51 | 69.01 ± 3.17 |
| Top5 | 5% | 86.41 ± 0.60 | 87.07 ± 0.90 |
| Top5 | 15% | 81.27 ± 1.32 | 82.04 ± 2.49 |
| Top5 | 30% | 76.57 ± 1.41 | 78.23 ± 3.34 |

6개 조건 모두 DNN의 평균 정확도와 평균 macro-F1이 높았습니다. 이는 반복 마스크에 대한 기술적 요약 결과입니다. [실행별 기록](experiment_results/robustness_all_runs.csv) · [요약](experiment_results/robustness_summary.csv)

### 6.3 결측 집중도와 새로운 입력 프로필

- **동일 총량:** 780개 셀을 가렸을 때 DNN 평균 정확도는 All13 **85.86% ± 1.81%p**, Top5 집중 결측 **82.93% ± 0.84%p**였습니다. 손상되는 특성과 결측 집중도가 함께 달라지는 비교입니다. [결과](experiment_results/matched_budget_all_runs.csv)
- **입력 프로필 그룹 검증:** 동일 입력 프로필을 한쪽에만 배치한 별도 학습 846행·테스트 359행 분할입니다. 그룹 CV로 선택한 Logistic Regression의 무결측 데이터 테스트 정확도는 **69.64%, macro-F1은 0.6111**이었습니다. 분할과 모델이 모두 기존 기준과 달라, 차이 전체를 입력 프로필 중복 탓으로 해석할 수는 없습니다. [결과](experiment_results/profile_group_test.csv)

## 7 설치 및 실행

기록된 실행 환경은 **Windows, Python 3.13.2, CPU PyTorch**입니다. 이 디렉터리에서 기록된 의존성을 설치하고 JupyterLab을 실행합니다.

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

`students_adaptability_level_online_education.csv`를 `analysis.ipynb`와 같은 위치에 놓고 노트북의 셀을 순서대로 모두 실행합니다. 노트북은 1,205행·14열 구조를 확인하고, 새 결과를 `experiment_results/`에 저장합니다. 저장된 노트북 출력과 CSV를 직접 살펴볼 수도 있습니다.

## 8 주요 디렉터리 구조

```text
.
├── README.md
├── README.ko.md
├── README.zh-CN.md
├── analysis.ipynb
├── requirements.txt
├── students_adaptability_level_online_education.csv
├── .gitignore
└── experiment_results/
    ├── clean_test_scores.csv
    ├── primary_classification_comparison.csv
    ├── primary_paired_bootstrap.csv
    ├── robustness_all_runs.csv
    ├── robustness_summary.csv
    ├── matched_budget_all_runs.csv
    ├── profile_group_test.csv
    ├── top5_selection.csv
    ├── All13/imputation_quality.csv
    └── figures/
```

## 9 한계 및 개선 방향

- 합성 MCAR 마스크는 실제 결측을 단순화합니다. 향후 MAR/MNAR 패턴과 독립된 출처의 데이터를 평가할 수 있습니다.
- 반복 마스크의 표준편차는 독립 데이터나 다른 분할의 불확실성을 나타내지 않습니다. Bootstrap 구간은 적합된 모델·분할·주 마스크에 조건부이며, 반복 프로필도 행 독립성에 제약을 줍니다.
- 같은 특성별 결측률에서 All13과 Top5의 결측 셀 수는 다릅니다. 동일 총량 비교도 결측 집중도까지 통제하지는 않습니다.
- Top5는 선택 모델과 내부 검증 표본에 의존합니다. 범주형 연관성과 순열 중요도는 인과 효과를 나타내지 않습니다.
- 프로필 그룹 결과는 더 넓은 일반화 검증의 필요성을 보여줍니다. 12-trial 튜닝 범위는 제한적이며, 향후 nested CV와 더 다양한 대치 기준 모델을 비교할 수 있습니다.

## 10 결과물 및 참고 자료

[영문 노트북](analysis.ipynb) · [기록된 환경](experiment_results/environment.json) · [모델 선택 순위](experiment_results/cv_ranking.csv) · [Top5 선정](experiment_results/top5_selection.csv) · [기록된 검증](experiment_results/verification.json)

