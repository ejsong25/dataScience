데이터 설명
===============

전세 데이터셋 (jeonse_dataset.csv)
----------------
> Independent Variables
>   > * Road_condition (도로 상태)
>   > * Contract_area_m2 (계약 면적(㎡))
>   > * Contract_period (계약 기간)
>   > * Building_age (건축 연식)
>   > * 동(One-hot-Encoding)

> Target Variable
>   > * Deposit (보증금)

<br/>

월세 데이터 셋 (wolse_dataset.csv)
----------------
> Independent Variables
>   > * Road_condition (도로 상태)
>   > * Contract_area_m2 (계약 면적(㎡))
>   > * Deposit (보증금)
>   > * Contract_period (계약 기간)
>   > * Building_age (건축 연식)
>   > * 동(One-hot-Encoding)

> Target Variable
>   > Monthly_rent_bill (월세)

<br/>

데이터 분석 방법
===============
> Regression
>   > * 전세 데이터: 
>   >   > 특정 조건(도로상태, 면적, 계약기간, 방 개수, 건물연식)에 따른 전세 보증금 예측
>   > * 월세 데이터
>   >   > 특정 조건(도로상태, 면적, 계약기간, 방 개수, 건물연식)에 따른 월세 예측

> Classification
>   > * 전세 데이터:
>   >   > (전세 보증금을 특정 기준에 따라 분류)
>   >   >   > * very cheap: -inf ~ (mean - 1.5 * std)
>   >   >   > * cheap: (mean - 1.5 * std) ~ (mean - 0.5 * std)
>   >   >   > * appropriate: (mean - 0.5 * std) ~ (mean + 0.5 * std)
>   >   >   > * expensive: (mean + 0.5 * std) ~ (mean + 1.5 * std)
>   >   >   > * very expensive: (mean + 1.5 * std) ~ inf
>   > * 월세 데이터:
>   >   > (월세를 특정 기준에 따라 분류)
>   >   >   > * very cheap: -inf ~ (mean - 1.5 * std)
>   >   >   > * cheap: (mean - 1.5 * std) ~ (mean - 0.5 * std)
>   >   >   > * appropriate: (mean - 0.5 * std) ~ (mean + 0.5 * std)
>   >   >   > * expensive: (mean + 0.5 * std) ~ (mean + 1.5 * std)
>   >   >   > * very expensive: (mean + 1.5 * std) ~ inf

Open Source SW Contribution
===============
> Used Function definition (and description)
> Architecture
> Preprocessing
> Learning Model Training and Testing
>   > * Linear Regression
>   > * Decision Tree
> Evaluation
>   > * K-Fold Cross Validation
>   > * Hold-Out Method
>   > * Bootstrap Method
> Result
> Github URL: https://github.com/ejsong25/dataScience
