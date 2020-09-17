---
title: "Field Experiment"
categories: 
  - 마케팅
use_math: true
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true
toc: true
toc_sticky: true
toc_label: Table of contents

---


# Experiments and Generalized causal inference
Shadish, Cook & Campbell (2002) CH_1

이 책은 철학적으로 cause and effect에 대해서 말합니다. 

---

### Causal Relationship

- Cause: _inus_ condition(an `I`nsufficient but `N`ecessary parts of `U`nnecessary but `S`uffieceint conditions)  
실험을 하는 이유는 cause and effect를 찾고 명확히 알고자 하는 과정입니다.  
하지만 대부분의 사람들은 causal relationships을 직관적으로 이해하고 있습니다.   
제대로된 causal inference를 위해선 althernative explanation을 rule out할 수 있는지, (하나하나의 alternative explanation이 모든 cause를 말할 수 있음) 그중에 어떤게 outcome을 일으켰는지 찾는게 중요합니다.

- Effect: Difference between what did happen and what would have happened  
- Counterfactual: What would have happened to those same people if they simultaneously had note received treatment

Effect와 Counterfactual은 아주 밀접한 개념입니다. 
Effect는 이미 어떤일이 어떤사람에 대해서 일어나고 만약 이사람이 그 상황이 아니라 control 상황(일어나지 않은 상황)에 있으면 무슨일이 일어났을지에 대한 그 차이를 말합니다. 즉, 똑같은 세상 A버전, B버전이 있을 시, A에서만 한 일이 일어나면 그 후의 A세상과 B세상의 차이라 볼 수 있습니다. Counterfactual은 일이 일어나지 않은 B세상에 개한 개념입니다.

다시 정리하자면  
- 원인은 결과보다 시간적으로 앞서야 합니다.
- 원인과 결과는 공동으로 변화하여야 합니다.
- 결과는 원인변수에 의해서만 설명되어져야하며 다른변수들에 대한 설명은 배제 되어야 합니다.


- Causal description and Causal explanation
    - X causes y - causal descripption
    - why does x cause y? - causal explanation  
Causal description은 단순히 x가 y가 cause and effect를 말한다면, Causal explanation은 왜 
그런지, 어떤 mechanism이 있는지 까지 설명하는것입니다.


---
### 실험의 구분

- randomizaed experiment : 우리가 강제로 random assignment을 하게 할 수 있는 실험 상황입니다. (random assignment와 random sampling은 다른 말임을 주의!)

- quasi-experiment : 준실험이라 하며, assignment를 가져다 컨디션에 강제로 시킬순 있으나 랜덤하게 시킬순 없는 상황입니다.   
예로, 새로운 영업플랜의 효과를 알아보고자 할때, 영업사원을 랜덤하게 나눠서 같은 세일즈 오피스에서 반은 옛날플랜에 따라서 인센티브받고 반은 새플랜을 가지고 하라하면 반발감이 생길게 뻔합니다. 그래서 지역을 나눠서 서울지부는 기존플랜 미국지부는 새로운 플랜으로 진행하여 그 차이를 보는 거라 할 수 있습니다. 즉, 이걸하는  이유는 무작위할당을 할 수 없는 상황이기 때문입니다.

- natural experiment: 실험은 아니지만 자연적인 exogenous shock이 있어야 합니다. 
쉽게 말하자면 갑자기 뭐가 확 변할때 그 이전과 이후로 나눠서 이전은 control, 이후는 treatment로 실험을 하는 것과 동일 선상으로 봅니다.  
ex) 한 정책에 대해 before after -> did match등으로 보정을 해 validate하게 만듬  

### Validity 란

   - Internal validity: 내적 타당성, 즉 연구의 변수가 가설대로의 효과를 보이는지, 측정된 결과변수의 변화가 실험변수의 변화에 의해서 발생한 것인지 검증하는 거라 보면 됩니다.  
   
   계량보다는 CB와 같은곳에서 신경쓰는 개념입니다. 여러차례 다른 material로 실험을 replicate하여 Internal validity를 확보합니다.
    
   - Construct validity: 측정값이 정말 하고자 하는 연구의 변수들을 측정하는 것인지를 보는 것 입니다.
   예를들어 사람의 행복을 측정하고 싶다하면 사람의 마음이라 설문조사와 같은 방법으로 측정을 해야 할텐데 과연 이게 제대로 측정을 한 것인지에 대해서 고민해야 합니다. (이런경우 보통 변수에 대해 정의를 하고 측정을 한뒤 테스트를하여 판단 합니다. 또는 저널에 기재된 연구를 인용합니다.)  
   
   - External validity: 연구에서 사용된 특정한 상황뿐만이 아닌 여러 다른 상황에서 재현가능한지, 일반화 가능한지를 말합니다.



# 대체로 해롭지 않은 계량경제학 CH_2 

먼저 이책에서 말하는 실험은 실험실에서 무작위 배정 실험이 아닌 자연실험(natural experiments)으로서 자연스러운 상태에서 진행 되었으나 무작위 실험과 동일한 함의를 갖는 변화를 의미합니다. 저자인 Angrist는 와 Pischke는 이러한 자연실험에 기반한 인과관계, 사회과학 실증연구 방법론을 정리합니다.

특히 ch2에선 의학연구에서 사용되는 유형의 무작위 실험이 질문들에 접근하는 하나의 이상적인 벤치마크를 제공한다느 말의 의미를 논의합니다. 
# The Experimental Ideal 

## 이상적인 실험

### 선택편의 문제
병원은 사람들을 건강하게 만드는가? 라는 질문에 대한 답을 알고 싶을때, 실증연구를 하고자 하는 사람들이 택하는 자연스러운 접근법은 병원에 입원한 적이 있는 사람들과 없는 사람들의 건강상태를 비교하는 방법입니다. 
아래의 표는 좋지 않다는 응답이 1, 매우 좋다가 5 일때의 자료 입니다.

|그룹|표본크기|평균 건강상태|표준편차|
|---|---|---|---|
|Hospital|7.774|3.21|0.014|
|No Hospital|90,049|3.93|0.003|

평균값의 차이는 0.72 이고 $t$-통계량은 58.9 로서 매우 유의하게 양호합니다.  
이 걸 그대로 받아들인다면 병원에 가면 오히려 안좋다는 점을 알 수 있는데 이는 잘못된 해석입니다.  
선택편의의 문제를 일으켰습니다.  


$D_i$ 가 환자(subject) $i$가 치료를 받았는지(1) 안 받았는지(0)를 나타내고 $Y_i$가 outcome이라면  
$\begin{align}
\text{Potential outcome} = 
\begin{cases} Y_{1i}   & \text{if }D_i=1, \\
Y_{0i} & \text{if }D_i=0
\end{cases}
\end{align}$

우리가 관찰한 결과는(Observed outcome) $Y_i = Y_{0i} + (Y_{1i} - Y_{0i})D_i$ 입니다.

잠시 표로 뭐가 뭔지 정리해보자면

|Explanation|Mathematical expression|
|---|---|
|치료를 안받았을때의 효과(Potential outcome)|$Y_{0i}$|
|치료를 받았을때의 효과(Potential outcome)|$Y_{1i}$|
|처치(치료 받을지 안받을지)(Dummy indicator)|$D_i$|
|실제 치료 성과(Observed outcome)|$Y_i$|
|처치효과(treatment effect or causal effect)| $Y_{1i}-Y_{0i}$|

여기서 $Y_{1i} - Y_{0i}$ 는 causal effect 입니다.  
하지만 개인$i$에 대한 outcome 이므로 여러명의 사람들을 생각하면 expectation을 구해야 합니다.  

$E[Y_{1i}|D_i = 1] - E[Y_{0i}|D_i = 0] =
E[Y_{1i}|D_i = 1] - E[Y_{0i}|D_i = 1]  + (E[Y_{0i}|D_i = 1]  - E[Y_{0i}|D_i = 0] ) $ 

$\text{Our observation} = 
\text{Average treatment effect on the treated(ATT)} + \text{Selection bias}$


이 식을 다시 ATT기준으로 본다면, 

$E[Y_{1i}|D_i = 1] - E[Y_{0i}|D_i = 1] = E[Y_{1i}|D_i = 1] - E[Y_{0i}|D_i = 0] - (E[Y_{0i}|D_i = 1]  - E[Y_{0i}|D_i = 0] )$

$\text{Average treatment effect on the treated(ATT)} = 
\text{Our observation} - \text{Selection bias}$

선택편의의 문제는 Treatment group과 Control group간에 다른 characteristics가 있기 때문입니다. 이 예에선 
병원에 가는 사람들은($D_i = 1$) 그렇지 않은 사람들($D_i = 0$) 보다 건강하지 못했기 때문입니다. 즉, assignment가 random이 아닙니다.


### 무작위 배정은 선택편의 문제를 해결한다

Simple linear model로 보자면,
$Y = \alpha + \beta X + \epsilon$ 에서 $X = D_i$일때 간단히 그냥  $\beta$값을 구한다면 $\beta$값은 biased 될 수 밖에 없습니다. 왜냐면 $D_i$가 random이 아니고 endogenous 하기 때문입니다. 
endogenous 하다? Error term($\epsilon$)중 underlying한 개인의 특성이 $D_i$와 correlated 하는 상황을 뜻합니다. 쉽게 말하자면 설명변수와 오차항의 상관이 0 이 아닌 상황을 말합니다. 이럴경우 우리가 측정하고자 하는 효과를 정확히 측정할 수가 없습니다. 

병원 예제로 보자면 치료의 효과가 더 적게 나오게 될 것 입니다.

이러한 문제를 Random assignment로 해결이 가능합니다.  
Random assignment는 집단의 크기만 충분히 크다면 대수의 법칙(Law of Large Numbers, LLN) 이라는 통계적 특성 덕에 Selection bias가 사라지기 때문입니다. 
$D_i$가 무작위 배정된 경우 
$E[Y_{0i}|D_i = 1]  - E[Y_{0i}|D_i = 0] = 0$ 이 성립하고 처치 상태별 기댓값의 차이는 처치의 인과효과를 포착합니다.


### 요약

무작위 배정으로 선택편의를 제거하면 뭐가 좋을까? 선택편의는 회귀식의 오차항(error term) 과 설명변수(regressor=회귀변수=독립변수=iv)간의 상관관계에 해당한다. 이러한 상관관계를 제거하는것이 회귀분석에서 인과효과를 추정할 수 있게 해준다.   
또한, 무작위할당 실험이나 다른 연구 디자인으로 얻은 데이터를 분석할때 처치집단과 통제집단이 실제로 비슷하게 보이는지 균형상태점검(Checking for balance) 과정을 거친다 (e.g. 각 표본 평균을 비교하는것)

이어서 Field Experiment에서 Random assignment를 쓴 논문을 예시로 마치겠습니다. 

# Do Targeted Discount Offers Serve as Advertising? Evidence from 70 Field Experiments?
Navdeep S. Sahni, Dan Zou, Pradeep K. Chintagunta (2017)
