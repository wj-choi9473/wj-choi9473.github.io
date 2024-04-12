---
title: "[Causal Inference] 논문을 통해 A/B test 결과들로 인사이트 도출 하는법 톺아보기"
categories: 
  - Data Science
tags:
  - Causal Inference
  - Experiment
  - Paper
  - Quantitative Marketing

last_modified_at: 2023-03-19
redirect_from:
    - /Data Science/[Causal Inference] 논문을 통해 A/B test 결과들로 인사이트 도출 하는법 톺아보기
---

기업에서 성과측정을 할 때 AB test를 주로 사용합니다. 여기에 더 나아가 여러번의 AB test 결과들을 가지고 인사이트를 도출할 수 있고, heterogeneous treatment effect를 계산해 personalization에 사용할 수 도 있습니다. 

이번 글에선 어떻게 더 인사이트를 도출하는지 논문을 통해 알아보겠습니다. 

Do Targeted Discount Offers Serve as Advertising? Evidence from 70 Field Experiments
Navdeep S. Sahni, Dan Zou, Pradeep K. Chintagunta (2017)

해당 논문의 저자들을 각 고객들을 타게팅한 이메일 할인쿠폰이 광고의 효과도 있다는것을 분석을 인사이트를 도출 했습니다.

저자들의 리서치퀘스쳔은 다음과 같습니다.

- Authors Research Questions 
    - Can targeted discount offers be profitable for a firm?
    - Can the offers serve as advertising messages?


# Introduction

- E-mail marketing
    - Vehicles for promotional discounts or coupons for price discrimination
    - Targeted discount offer as an advertising

- May cause selection bias because of nature of E-mail marketing
    - Consumer who opt-in and opt-out might be different
    - It is targeted

우선 이메일 마케팅은 보통 일반적으로 가격차별을 위해 할인쿠폰을 발급하기 위한 수단으로 쓰일 수 있습니다.
저자들은 여기에 더해서 광고적인 효과도 있다고 합니다. 즉, 다음 구매에 이 회사의 제품으로 구매하게 하는 효과를 볼 수 가 있습니다.   

이러한 결론을 도출 하기 위해 단순히 이메일 마케팅 보낸 그룹과 아닌 그룹을 비교해서 분석할 순 없습니다. 이메일 마케팅 특성상 selection bias가 있기 때문입니다. 전체 집단을 보려하면, `이메일 마케팅 수신 동의를 한 소비자와 안한 소비자의 특성`이 다를 수 있고, `이메일 수신을 동의한 소비자들 사이`에서도 자주 구매하는지 등 행동 `특성에 따라 타게팅`을 다르게 할 수 있기 때문입니다.   

# Empirical Setting and Data

<img src = "/assets/images/posts/datascience/abtestinsight/1.png">{: .align-center}
저자들은 전체적인 소비자를 본것이 아닌 `마케팅 수신메일을 동의한 소비자들중` `targeted` 된 소비자들을 랜덤하게 홍보메일을 보내는 그룹과 안보내는 그룹으로 나누어진 70개의 실험(즉, 프로모션 캠페인)을 분석했습니다.


<img src = "/assets/images/posts/datascience/abtestinsight/2.png">{: .align-center}

전체 샘플 집단과 옵트인한 그룹을 비교해 봤을때 표에 나와있는 expenditure and transactions per year 과 같은 특성들이 통계적으로 다르다고 합니다. 

<img src = "/assets/images/posts/datascience/abtestinsight/3.png">{: .align-center}
데이터를 더 자세히 보면, 프로모션 기간은 보통 이주에서 한달 사이에 다 끝났고 총 122개의 프로모션 캠페인중 76개의 캠페인이 treatment 와 control을 준 캠페인이고 그 중에서 6개의 캠페인은 참가된 개인의 수가 100명이 안되어 제외하고 100명 이상을 참여시킨 캠페인 총 70개의 캠페인 데이터를 가지고 분석을 진행했다고 합니다.
또한 피규어 2를 보시면 각 개인이 프로모션 안에 여러번 참여하게 될 수도 있다는 것을 보여줍니다.

<img src = "/assets/images/posts/datascience/abtestinsight/4.png">{: .align-center}
그 뒤, 정말 treatment group이랑 control group간 randomization이 잘 되었는지 체크하기 위해
두가지 점검을 하는데요  
처음엔 여러 특성들을 비교검정를 합니다 총 71445 개의 instance중(개인이 중복거래도 포함) 61377이 프로모션캠페인이메일을 받고 10068이 컨트롤 그룹으로 되었습니다.   
각각의 특성들을 비교 하고 동일하다는 귀무가설을 거부할 수가 없어 두 그룹은 같다고 가정하며  

두번째 점검으로 각 캠페인에 대한 dummy indicator를 넣어 70개 캠페인의 고졍효과로 회귀분석을 실행했을시에도 인구통계나 과거transaction이 통계적으로 유의한 계수는 없는걸 확인해 실험에 참가한 각 개인은 무작위 할당이라는 기업의 주장(저자들의 가정)을 뒷받침 합니다.

<img src = "/assets/images/posts/datascience/abtestinsight/5.png">{: .align-center}
그 외에도 테이블 4를 보시면 회사가 각 개인별로 RFM 을 측정해 타겟팅을 했다는 증거와  
(Rfm은 r recency 마지막거래로부터 날짜 days f 는 frequency 오퍼전에 얼마나 많은 거래를 했는지 m은 monetary로 돈을 얼마나 썻는지를 나타냄)  

테이블 6을 보면 Redemption rate, 즉 오퍼받은 사람들중 얼마나 그 프로모션을 썻는지에 대한 비율이 0.1%대로 굉장히 작다는걸 알 수 있습니다.  

또한, 피규어 3을 보시면 평균지출의 분포가 오퍼를 받은 그룹의 spending 분포가 더 돈을 많이 썻다는걸 볼 수 있습니다. 

이렇게 간단히 데이터 분석을 한 이후 저자들의 질문의 대한 답을 알아가기 위한 단계로 갑니다.

## Empirical Approach

우선 empirical approach를 하기에 앞서 저자들은 플랫폼이 운영하는 실험에 참여한 적이 없는 개인에게 이러한 프로모션이 미치는 여향은 추정 못한다고 하였습니다. 어찌보면 당연한 이야기지요.  

저자들이 보고자 하는것은 프로모션을 받은 개인의 개인이 프로모션 이메일을 받았을때와 안받았을때의 차의 평균 지출을 보고자 합니다.   
수식으로 표현 하자면  
 
$\theta \equiv E_{ij}(Y_{ij}^1 - Y_{ij}^0 \mid D_{ij} = 1)$ 입니다.

- $\theta$ : Average treatment effect on the treated population
- $i$ : Individual users indexing
- $j$ : Experiment indexing
- $Y_{ij}^1 - Y_{ij}^0$ : Expenditures for $i$ in the presence and the absence of an offer
- $D_{ij}$ : Dummy indicator wheter $i$ is a part of experiment $j$

<img src = "/assets/images/posts/datascience/abtestinsight/6.png">{: .align-center}  
두번째 포인트론 각 실험마다 treatment 그룹에 할당되는 개인의 비율이 다르다고 하였습니다. 
피규어4를 보시면 70개의 실험에서 treatment propensity (프로모션을 받을 확률 정도로 생각하셔도 됩니다.) 가 0.31 에서 0.97까지 크게 다른데 회사의 의사결정 때문일것이라고 합니다.
왜냐면 사람들이 오퍼를 잘 받아들일 수도 있는데 안보내서 판매를 못하는 기회비용이 크기 때문입니다. 

그래서 데이터의 이러한 변동으로 인해 모든 실험 j에 대해 별도의 처리 효과 (θj)를 추정하여 θ를 추정하고 샘플 가중 평균 효과를 계산합니다.

모든 실험 j는 무작위 실험이기 때문에 처리 군과 대조군을 비교하면 실험 제안 j를 대상으로 한 집단에 대한 실험 별 처리 효과의 편견없는 추정치를 얻을 수 있습니다.

$\theta_j \equiv E_{i}(Y_{ij}^1 - Y_{ij}^0 \mid D_{ij} = 1)$  
$\theta_j$ 의 sampling distribution이 정규분포라 할때, $\theta_j \sim N(\hat\theta_j, \hat\sigma^2_j)$  
$\theta =\dfrac{\sum n_j\theta_j} {\sum n_j} \sim N \left( \dfrac{\sum n_j\hat\theta_j} {\sum n_j}, \dfrac{\sum n^2_j\hat\sigma^2_j} {(\sum n_j)^2}\right) $  
$n_j$ : number of individuals in experiment j

## Results

Empirical appporach 에서 접근한 방식대로 샘플데이터에 대해서 회귀분석을 합니다.

$y_{ij} = \alpha_j + \theta_{j}T_{ij} + \epsilon_{ij}$
- $\theta_j$ : For all 𝑗 are estimated jointly and the standard errors are robust, clustered by individual
- $T_{ij}$ : Wheter individual $i$ was in the treatment group for experimental campaign $j$  

첫째로는, 오퍼 즉 프로모션이 유효했던 기간 동안 프로모션이  개인의 지출에 미치는 영향에 을 알아보려합니다.

즉, 오퍼가 플랫폼에서 개인에게 구매를 유도할 수 있다면, 오퍼를 안받는 개인에 비해 제안을 받는 개인에 대한 지출이 증가하는 것을 보아야 합니다.

<img src = "/assets/images/posts/datascience/abtestinsight/7.png">{: .align-center}

테이블 7을 보시면 결과를 확인할 수 있는데요  
오퍼가 유효할때 타겟된 개인들에게 평균 3.03 의 지출을 더 하게 하고 퍼센트로 따지자면 37% 의 지출을 늘리는것으로 확인이 됩니다. 

즉, 저자들의 첫번째 리서치 퀘스쳔에 대한 답을 알 수 있었습니다.


<img src = "/assets/images/posts/datascience/abtestinsight/8.png">{: .align-center}
테이블 9는 방금전의 식에 개인의 rfm 특성을 넣은 회귀분석 결과로 
여기서 주목할 점은,  
최근에 거래하지 않은 개인에게 큰 영향을 미침 -> 2번째 리서치 퀘스쳔의 대한 답으로 볼 수 있는 가격차별요소뿐만 아닌  reminder effect를 볼 수 있다는 것 입니다.

<img src = "/assets/images/posts/datascience/abtestinsight/9.png">{: .align-center}
피규어 5는 결과를 시각화 해서 보여준것인데요 

타게팅될 확률과 오퍼의 효과는 모두 f와 m에 따라 증가합니다. 즉, 웹 사이트에서 더 많,더 자주 지출 한 개인은 제안에 더 많이 반응하고 회사는 제안으로 이들을 목표로 삼을 가능성이 더 높습니다.

그러나 타게팅 전략과 오퍼 효과 사이의 이러한 일관성은 r에 없습니다. 제안은 1 사분 위와 2 사 분위 (비교적으로 최근에 웹 사이트에서 거래 한 사람)의 개인에게 가장 효과적이지 않습니다.

하지만 회사는 2 사 분위의 개인에게 제안을 보낼 가능성이 가장 높습니다. 방향적으로 회사는 최근 거래자들에게 제안을 보내지 않는 전략을 따르는 것처럼 보이지만, 이 시각화는 마지막 거래 이후 일수 기준으로 중앙값을 넘는, 즉 구매한지 오래된 소비자에게 더 많은 제안을 보내면 더 많은 혜택을 얻을 수 있음을 알려줍니다. 

지금까지의 결과를 보면 프로모션이 가격차별도 있지만 광고고효과인걸 볼 수 있는데요


 두번째 분석으로 저자들은 두번째 리서치 퀘스쳔이자 첫번째 결과에서 본 광고효과에 대해 측정을 합니다. 

 오퍼가 광고효과, 즉 플랫폼에서 이용할 수 있는 서비스를 상기시키는 역할을 하면, 그 제안이 발송된 시점부터 경과한 시간에 따라 그 영향이 감소 할 수 있다고 합니다. 

이를 확인하기위해 저자들은 프로모션이 만료된 후에도 개인이 더 많은 지출을 하는지 프로모션이 만료된 후 트리트먼트와 컨트롤 조건에서 개인에 대한 지출을 비교합니다. 

$\tilde Y_{ij} = \alpha_j + \tilde \theta_{j}T_{ij} + \epsilon_{ij} $  
$\tilde Y_{ij}$ :$i$’s total expenditure during a period after offer $j$ expires 

<img src = "/assets/images/posts/datascience/abtestinsight/10.png">{: .align-center}
피규어 6을 보시면 결과로 프로모션이 만료된 후에도 할인을 받을 수 없는데 지출이 증가하는걸 볼 수 있습니다.

그렇지만 소비자들이 정말 오퍼를 받음으로 기억을 해 나중에 구매하는지는 정확히 알 수 없지만 개인의 carryover effects는 확인 할 수 있는데요, 그래서 다시 프로모션 기간에 구매를 했는지 더미변수를 넣어 확인하였고 테이블10의 결과 표를 보시면 프로모션이 유효한 동안 거래 한 개인이 거래를했지만 오퍼을받지 못한 개인보다 제안 후 주에 지출 할 가능성이 더 높다고 저자들은 해석했습니다.
<img src = "/assets/images/posts/datascience/abtestinsight/11.png">{: .align-center}

그 다음으로 이메일이 전송된 후 시간이 지남에 따라 오퍼의 효과가 어떻게 감소하는지를 살펴보았는데요, 
프로모션 오퍼가 발송 된 후 3개월 동안 매주 트리트먼트 그룹과 컨트롤그룹간의 지출 차이를 추정합니다.
피규어 7은 그 추정을 시각화 하였는데요, 처음 4 주 동안 트리트먼트 그룹이 컨트롤 그룹보다 더 많이 지출 하는걸 볼 수 있습니다(3주차 제외). 
<img src = "/assets/images/posts/datascience/abtestinsight/12.png">{: .align-center}

그 다음으론  오퍼의 종류가 다른 섹션의 티켓 판매에도 영향을 미치는지를 조사하는데요, 
언급하진 않았지만 데이터는 스텁허브 같인 티켓리셀링 플랫폼에서 얻은 데이터 인데요, 할인 프로모션에서 대부분 MLB티켓은 할인에서 제외를 한 오퍼가 많았습니다.

만약 오퍼가 다른 섹션의 티켓 판매에도 광고효과로 긍정적인 영향을 보인다면 MLB의 티켓 구매도 증가 할 수있다고 볼 수 있습니다. 테이블 11을 보신다면, 평균적으로 4.04 달러의 지출이 증가하고 mlb 티켓이 오퍼가 제외되어도 그냥 지출도 증가하고 전에 구매한 이력이 있다면 지출이 더 증가 한걸 볼 수 있습니다.

그러므로 저자들은 두번째 리서치 퀘스쳔인 광고효과가 있는걸 보여주는걸로 마무리를 합니다.  

$y_{ij} = \alpha_j + \delta_{j}T_{ij} \times PastMLB_{ij} + \gamma_{j}PastMLD_ij + \epsilon_ij$


<img src = "/assets/images/posts/datascience/abtestinsight/13.png">{: .align-center}

