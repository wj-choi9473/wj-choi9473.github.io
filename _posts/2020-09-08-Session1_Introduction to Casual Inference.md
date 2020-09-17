---
title: "Introduction to Causal Inference"
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

이 포스팅은 카이스트 경영대학 윤태중 교수님의 마케팅 모형론 수업을 들으며 정리한 것입니다.

사설 강의/책 에서 데이터 분석/데이터 사이언스라고 하며 주로 가르치는건 프로그래밍, 분류/예측, 텍스트마이닝 등등 멋져보이고 재미있어 보이는걸 가르쳐 줍니다. (물론 이런것도 당연히 중요하다고 생각합니다) 하지만 그전에 먼저 알아두어야 하고, 실제 일을 할 때, 의사 결정을 위해 더 중요하고 필요하다고 생각하는건 인과 관계를 제대로 설명하는것 이라고 생각하게 되었습니다. (머신러닝과 방향성이 좀 다르긴 하지만!)

데이터분석은 인과 관계를 바탕으로 스토리텔링 하는 것이고 데이터에 기반한 분석의 힘을 실어 주으려면 개연성이 필요합니다. 즉, 납득할 수 있는 명확한 인과관계를 보여주는 것이 연구자로서 혹은 데이터 분석가로서 중요하다고 생각하게 되었습니다.  

# Causal Inference 란?

인과 관계추론 이라 하며 마케팅 액션이 있으면 그 효과를 판단하기 위해서 measure 하는것 또는 원인 파악을 위한 것입니다. 
- 원인 찾기 ex) 왜 갑자기 DAU가 떨어졌지?
- 효과 측정하기 ex) 정책의 효과 

최근에 핫한 이슈인 공공의대 정책을 예로 과연 지방의료의 현실이 좋아질것이냐 아닐것이냐 를 볼때(그저 예시 입니다!)  

input - 지방에 의사 확대 outcome - 효과 측정을 위한 measure을 정하고 (지방에 분만율이 어떻게 되느냐, 지방 환자의 사망률이 낮아지느냐 등 도메인 전문가와 상의해서 정해야겠지요? ) 
이런 인과 추론을 하는데 있어서 실험실이 아닌 실생활이라 딱 공공의대 정핵의 순수한 효과만 보고 싶지만 경제사정이라던가 지방에 병원의 숫자도 달라질수도 있고 이러한 각종 요인들이 outcome에 영향을 주기 때문에 이러한 요인을 최대한 제거하고 봐야합니다.  




# Case Study

## Promotional E-mail Frequency

디지털 마케팅 부서장은 제품담당 팀에서 소비자들에게 너무 많은 메일 보내는게 고민입니다. 광고성 이메일을 고객에게 보내는 빈도수를 설문조사에 따르면 고객들은 이메일을 너무 많이 받고 있다고 하고, 이멜을 많이 받을 수록 campaign clickrate이 낮아지는 자료를 찾아서 의견을 냈습니다.

|All consumers|Result|
|---|---|
|메일 너무 많이 받음|72%|
|메일 적당히 받음|21%|
|메일 별로 안받음|7%|

|top 25% 매출을 내는 consumer|Result|
|---|---|
|메일 너무 많이 받음|82%|
|메일 적당히 받음|12%|
|메일 별로 안받음|5%|

|bottop 25% 매출을 내는 consumer|Result|
|---|---|
|메일 너무 많이 받음|32%|
|메일 적당히 받음|56%|
|메일 별로 안받음|12%|

반대로 스포츠용품 부서장은 광고성 이메일 빈도수를 규제하지 않고 그대로 하자는 주장을 준비하기로 했습니다. 그는 자신의 주장을 뒷받침할 증거가 있는지 과거 구매 자료를 조사하라고 팀에 요청을 했습니다. 특히 최근 12개월 동안 고객들의 주문 건수, 최근 12개월 동안의 총 달러 매출, 지난 주문 이후 시간 등 주요 지표와 관련된 이메일 빈도가 어떻게 연관되어 있는지 증거를 보고 싶어했습니다. 조사 결과 다음과 같은 분석결과를 가지고 이메일 빈도가 높을수록 주문건수, 매출, 지난 주문 이후 시간이 짧다는 주장을 했습니다.

||Number of orders during last 12 months|
|---|---|
|Avg. weekly e-mail freq: 0 - 1.9|1.7|
|Avg. weekly e-mail freq: 2 - 3.9|4.8|
|Avg. weekly e-mail freq: 4 or more|7.3|

||Total sales during last 12 months|
|---|---|
|Avg. weekly e-mail freq: 0 - 1.9|\$51|
|Avg. weekly e-mail freq: 2 - 3.9|\$254|
|Avg. weekly e-mail freq: 4 or more|\$518|

||Time since last order|
|---|---|
|Avg. weekly e-mail freq: 0 - 1.9|8.1 months|
|Avg. weekly e-mail freq: 2 - 3.9|4.2 months|
|Avg. weekly e-mail freq: 4 or more|1.8 months|

### Interpretation is as inportant as analysis

위 케이스는 두가지 문제를 가지고 있습니다.
1. 광고성 메일이 소비자에 따라 어떻게 달라지는지, 다들 똑같은지 모릅니다.

뭐가 원인이고 뭐가 outcome인지 생각해보자면 

마케팅 부서장은 이메일을 얼마나 많이 보내느냐에 따라 outcome이 소비자의 불만족이 될것이라 판단을 합니다.  
스포츠용품 부서장은 원인과 결과를 서로 반대로 놓고 해석을 하였습니다.  
스포츠용품 부서장은 email frequency가 원인, 구매가 outcome이라 했지만 조금만 생각해 보면 스포츠용품 부서장은 `reverse causality`를 범한걸 알 수 있습니다. 왜냐하면 이메일을 많이 받는 그룹은 일반적으로 이메일을 보낼 시 소비자를 타게팅(RFM모형 등을 이용)하는데 잠시만 생각해보면 물건을 살 수록 프로모션 메일이 많이 보내고 많은 혹은 비싼 물건을 구입한 고객일 수록 이멜을 많이 받는것이기 때문입니다.  


2. 마케팅 부서장이 모은 survey data는 `sample selection`을 하는데에 문제가 있습니다.

현재 고객들을 대상으로 설문을 진행하였는데 이미 이메일을 많이 보내고 있는 시점이라 이메일에 짜증이 딱히 안나는 고객들만 남고 예민한 고객은 스팸차단, 이탈등을 했을 수도 있습니다. 그렇기에 실제 불만보다 적을 수도 있고 모르는 상황이 발생합니다. 즉 대표성을 못띄는 `selection bias`가 있다고 볼 수 있습니다.  


---

또한 소비자의 불만이 매출의 감소로 이어지는가?란 질문에 대한 답도 확실히 알 수 없습니다.
과연 광고성 이메일에 frequency 가 매출에 영향이 있을까? 이것 또한 알 수 없습니다.

---



## 비즈니스에서 좋은 분석이 어려운 이유

### Fundamental Conflict
- 비즈니스는 이익을 극대화하기위해 targeted decisions을 내립니다.
    - 예로 광고는 더 살 확률이 높은 소비자에게 노출을 합니다
- 분석의 gold standard는 experimentation 입니다.
    - 그룹별 targeted가 아닌 randomly assigned 되야 합니다.
    
데이터분석을 위한 프로그래밍 능력이 아무리 좋아도 casual inference를 알아야 제대로 해석할 수 있습니다.  
우리가 가지고 있는 데이터는 대부분 실험을 통한 데이터가 아닙니다.  
예로, 광고는 targeted advertising을 합니다. 문제점은 여기서 나온 데이터로 타겟 광고를 받은 사람과 받지 않은 사람 두군을 나눠 구매비율을 비교하면 targeted 그룹이 이미 그 물건을 살 확률이 높으니 (여러 특징을 보고 살 확률이 높은 소비자를 타게팅 해서 광고를 보여주니) 단순히 그룹을 비교하기는 어렵습니다. 그래서 일반적으로 회사 데이터는 이러한 transaction으로 이루어진 데이터라 해석하는데 큰 어려움들이 발생합니다.  

그래서 casual inference가 기본이 되어 정확한 해석이 필요합니다.  

---
### Data-generating process

현실에서는 우리가 가지고 있는 데이터가 실험에서 얻은게 아니기에 selection 문제가 있습니다. 
그래서 data generating process, 즉 데이터가 어떻게 생성이 되었는지도 유심히 봐야합니다. (random assignment인지 아닌지 등)

#### check-list

-  Are there pre-existing differences between groups?
    - i.e. “Could groups be probabilistically not equivalent?”
- Is there a common driver of both managerial decisions and outcomes?
    - i.e. “Are decisions and outcomes both responding to a common catalyst?”
- Is there reverse causality?
    - i.e. “Did managerial decisions cause outcomes to change, or was it outcomes that spurred managerial decisions?”
- Is there a plausible coincidence that could explain the outcome?
    - i.e. “What else might be going on?”
    
모든 질문에 no라고 답할수 있어야 합니다.

---
### 원인파악/ 효과측정을 위해 제일 좋은 방법은?
#### `Randomized Control Test`
그렇다면 어떤식으로 해야 이 문제를 풀 수 있을까요?
모든 변수를 통제하고 email frequency만 변화를 모든사람들이 같은 갯수의 이메일을 받고 불만을 측정하면 되겠지요. 즉, 측정 하고자 하는 변수 이외에는 모든 것들을 고정시키고, 확인하고 싶은 항목만 변경해서 테스트를 하면 됩니다.   
하지만 현실은 녹록치 않아서 다른 방법론이 많이 있습니다. 앞으로 배워갈 예정입니다 DID라던가...Matching Methods라던가...  


DID의 개념을 간략히 설명하자면 완전한 실험을 못하니 준실험적 방법중 하나라 알 수 있습니다.  
두 집단의 difference 를 difference한 두 시기로 비교해보는 것입니다.   

예로 subscription에 가입하게하는 프로모션을 통해 소비자의 LTV를 측정한다 하였을때 어떻게 효과를 알 수 있을까를 보자면

- Pick2geographicregions(byZIP,regionalsalesoffice,etc.)
- Pick2timeperiods(byweek,month,etc.)

||Region1|Region2|
|:---|---:|---:|
|Period1|Outcome in Region1 <br>before test|Outcome in Region2 <br>before test|
|Period2|Outcome in Region1 <br>before test|Outcome in Region2 <br>before test|

이렇게 나누어서 측정을 한 뒤

||Region1|Region2|
|:---|---:|---:|
|Period1|Control Group1 <br>Avg.LTV \$350|Control Group2 <br>Avg.LTV \$390|
|Period2|Control Group3 <br>Avg.LTV \$320|Target Group <br>Avg.LTV \$400|

- seasonal effect로 $30 (350 - 320)의 차이  

- regions 에 따른 차이(각 객체의 특성)로 $40 (350 - 390)

차이들이 있다면 이를 제거하면 순수한 프로모션틀 통한 LTV의 outcome을 알 수 있을 것입니다.

즉, seasonal effect의 차이를 제거하기 위해 (Target group-control group2)-(control group3-control group1) = 10 -(-30) = $40 입니다.  

또한, regions에 따른 차이를 제거하기 위해 (Target Group - Control Group3)-(Control Group2 - Control Group1) = 80 - 40 = $40 입니다.




### 공부 및 참고 하면 좋을거 같은 자료들 

Causal Inference 에 대한 자료를 찾다보니 카카오 데이터 분석가 이민호님의 [발표자료](https://www.slideshare.net/lumiamitie/causal-inference-primer-20190601?ref=https://lovablebaby1015.wordpress.com/2019/06/04/causal-inference-%EC%9D%B8%EA%B3%BC%EA%B4%80%EA%B3%84%EC%B6%94%EB%A1%A0/)도 발견했는데 인사이트를 얻기 좋은거 같습니다.  

또한,  NC소프트 데이터분석팀의 [기술블로그](https://danbi-ncsoft.github.io/Posts/)에서 인과관계를 찾아서란 시리즈를 발견했습니다. 그 외에도 좋은 인사이트가 많습니다. 

공부하고 읽어 보기 좋은 책들
- 고수들의 계량경제학 (대체로 해롭지 않은 계량경제학의 쉬운버전 입니다)
- 대체로 해롭지 않은 계량경제학


```python

```
