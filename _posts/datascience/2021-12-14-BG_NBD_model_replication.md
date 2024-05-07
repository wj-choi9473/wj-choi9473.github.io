---
title: "[Quantitative Marketing] 고객 생애 가치(Customer Lifetime Value)란?"

categories: 
  - Data Science
tags:
  - Paper
  - Quantitative Marketing

last_modified_at: 2022-04-19
redirect_from:
    - /Data Science/[Quantitative Marketing] 고객 생애 가치(Customer Lifetime Value)란?

---

기업은 다양한 이유로 고객 한 명당의 현재 및 미래 가치를 정량화합니다. 예를 들자면 `고객 획득 비용을 평가`하여 `해당 활동에 얼만큼의 금액`을 쓸것인지, 현재 기업의 `생애 가치가 높은 고객이 많은지, 낮은 고객들이 주인지`, `어떤 고객을 대상`으로 마케팅 켐페인 할지 (타겟팅), 특정 고객을 `타게팅 할 때 얼마를 지출`해야하는지, `고객의 가치를 높이는 활동이 정말 가치를 높이는지`에 대해 인사이트를 얻기 위함 입니다. 즉, 지표수립 부터 실제 마케팅 활동까지 다양하게 쓰입니다. 

>고객 생애 가치란 그래서 무엇이고 어떻게 계산 할까요?

# Customer Life time value(CLV) 란 무엇이고 어떻게 구할까?

CLV 란 소비자가 기업과 거래를 시작한 다음 이를 멈추기 까지 기업에 얼마만큼의 이익을 가져올것인가를 예측한 값입니다.
다시 표현하자면, "현재" 고객이 미래에 가져다 주는 수익을 고려하는것이며, future purchaing patterns of customers 를 예측하는 것입니다.

Individual-level에서 future buying behavior을 알려면 어떻게 할까요?

기본적으로 CLV 모델의 기본적 구성요소는 3 가지 입니다.
- 개별 고객이 거래하는 빈도(Frequency) 
- 개별 고객이 거래하는 규모(Monetary)
- 개별 고객의 고객 관계 예상 기간 (age)

> 기본적 구성요소에 맞게 모델링을 할 때 고려해야할 사항들이 있습니다.

1. Contractual setting (계약형;정기구독과 같은 서비스) 인지 Non-contractual setting (비계약형;단순 쇼핑) 인지
    - Contractual setting에선 customer churn을 observe할 수 있지만 Non-contactual상황에선 불가능합니다.
    - 예를들어 고객이 통상 2주에 한번 마트에서 장을 보는데 4주동안 구매가 없다면, 이탈했다고 봐야할까요? 아니면 단지 고객의 일반적인 변동성의 일부일까요?
    - Non-contractual때는 이러한 고민을 위해 측정 시점에서 고객이 계속 고객으로 남아있을 확률을 계산합니다. 

2. 또한 구매가 discrete(e.g.티켓)인지 continuous(e.g. 여러 물건 구매) 인지도 고려해야 합니다.

예시와 함께 표로 정리하자면 다음과 같이 고려해야할 케이스를 나눌 수 있습니다.

|Value/Setting  | Non-contractual | Contractual |
| ---- | ---- | ---- | 
|Continous| 마트 쇼핑 | 쿠팡 와우멤버십, 아마존 audible membership | 
|Discrete| 영화티켓 | 월 또는 연 단위 구독 (통신,넷플릭스) |

Contractual 같은 경우 고객 관계 상태 (지속 또는 종료) 를 관측할 수 있으므로 비교적 모델링이 간단합니다. `관계를 종료한 고객`과 `아닌 고객의` 데이터셋을 가지고 `예측의 문제`로 풀 수 있기 떄문입니다.

# Non-contractual Setting: BG/NBD and Gamma-Gamma 

Non-contractual Setting 에서는 CLV을 구하는 잘 알려진 방법은 다음과 같습니다.  

 `CLV = BG/NBD * Gamma-Gamma`
- BG/NBD (Beta Geometric/Negative Binomial Distribution) 확률 모델: 미래의 예상 기대 구매 횟수 
- Gamma-Gamma 확률 모델: 미래의 예상 기대 수익 

BG/NBD와 Gamma-Gamma 모델에 대한 페이퍼는 여기서 찾을 수 있습니다.

[“Counting Your Customers” the Easy Way: An Alternative to the Pareto/NBD Model](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) Fader et al. (2005)  

["The Gamma-Gamma model of monetary value"](http://www.brucehardie.com/notes/025/gamma_gamma.pdf) Fader et al. (2013)

또한, 해당 모델을 python 으로 구현해 놓은 패키지는 ["pymc-marketing"](https://github.com/pymc-labs/pymc-marketing ) 입니다.  
이번 포스팅에선 BG/NBD 모델 work shop의 일부분을 replicate 해보려 합니다. 다음에 시간이 될 때 Gamma-Gamma 모델도 같이 replicate하고  pymc-marketing 라이브러리로 비교 해보겠습니다. 

## BG/NBD Replication
 
BG/NBD를 replicate하기 위해 연구에 쓰인 dataset을 쓸것이며 [여기](http://brucehardie.com/notes/004/)에서 다운 받을 수 있습니다.

데이터 셋에 대한 설명은

"The worksheet Raw Data contains these data for a sample of 2357 CDNOW customers who made their first purchase at the web site during the first quarter of 1997. We have information on their repeat purchasing behavior up to the end of week 39 of 1997" 라고 나와 있습니다.

이 모델에선 3가지 information이 필요합니다.

- $x$ is the number of transactions from customer in the time period (0,T] (frequency-how many transactions he maid in a specified time period)  
고객이 반복 구매 한 횟수

- $t_x$ is the date of the most recent purchase in weeks since the customer’s first purchase (0<$t_x$) (recency - when his last transaction occurred)  
마지막 구매로 부터 지난 기간

- T is the time to be considered in weeks since the customer’s first purchase | The legnth of time over which we have observed his purchasing behavior.  
고객별 구매 행동을 관찰한 기간


```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('bgnbd.xls', sheet_name='Raw Data')
df.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>x</th>
      <th>t_x</th>
      <th>T</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>30.428571</td>
      <td>38.857143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1.714286</td>
      <td>38.857143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
    </tr>
  </tbody>
</table>
</div>



### BG/NBD Assumptions

고객의 이탈이 구매 후 바로 일어난다면 betageometric model로 이를 알 수 있으며 다음과 같은 가정을 합니다.

1. While active, the number of transactions made by a customer in time period of length $t$ follows is distributed  Poisson with mean transaction rate $\lambda t$  
고객이 이탈하지 않는 동안, 일정한 기간 $t$ 동안의 구매횟수는 Pois($\lambda t$) 을 따른다.  

2. Heterogeneity in transaction rate $\lambda$  between customers follows a gamma distribution with shape $r$ and scale $\alpha$  
고객마다 일정한 기간동안 구매하는 횟수는 다르고 이는 $\lambda \sim Gamma(r,\alpha)$ 를 따른다.

3. Each customer becomes inactive after each transaction with probability $p$ with geometric distribution.  
$P$(inactive immediately after $j$ th transaction) $= p(1-p)^{j-1}$, $j = 1,2,3...$  
고객은 구매 직후 일정한 확률 $p$ 로 이탈하고 (unobserved dropout propensity)

4. Heterogeneity in $p$ follows a beta distribution with shape parameters $a$ and $b$  
고객마다 다른 이탈 확률을 가진다. 

5. Transaction rate $\lambda$ and dropout probability $p$ vary independently between customers.  
구매 확률과 이탈 확률은 서로 영향을 주지 않는다.

### Model Development at the Individual level

> The likelihood a customer makes $x$ purchases in a given time period $T$ is :

$L(\lambda, p \mid X=x, T) = (1-p)^x \lambda^x e^{-\lambda T} + \delta_{x>0}p(1-p)^{x-1}\lambda^x e^{-\lambda t_x}.......(1) $

> 하지만 이건 $\lambda$ 와 $p$ 를 알아야 하지만 unobserved 이기에 구할수 없어서 purchase history $X = x,t_x,T$를 통해 다시 Likelihood fuction을 쓰자면:  

$L(r, \alpha, a, b \mid X=x, t_x, T) = A_1 A_2 (A_3 + \delta_{x>0} A_4).......(2)$

where
- $A_1 = \dfrac{\Gamma(r+x)\alpha^r}{\Gamma(r)}$
- $A_2 = \dfrac{\Gamma(a+b)\Gamma(b+x)}{\Gamma(b) + \Gamma(a+b+x)}$
- $A_3 = \dfrac{1}{\alpha + T}^{r+x}$
- $A_4 = \bigg( \dfrac{a}{b+x-1}\bigg)\bigg(\dfrac{1}{\alpha + t_x}\bigg)^{r+x}$

> Parameters 추정을 위해 식 2를 log-likelihood fuction으로 바꾸자면:

$\ln[L(r, \alpha, a, b \mid X=x, t_x, T) = \ln(A_1) + \ln(A_2) + \ln(e^{\ln(A_3)} + \delta_{x>0} e^{\ln(A_4)})].......(3) $

where

- $\ln(A_1) = \ln[\Gamma(r+x)] – \ln[\Gamma(r)] + r\ln(\alpha)$
- $\ln(A_2) = \ln[\Gamma(a+b)] + \ln[\Gamma(b+x)] – \ln[\Gamma(b)] –  \ln[\Gamma(a+b+x)]$
- $\ln(A_3) = -(r+x) \ln(\alpha + T)$
- $\ln(A_4) = \begin{cases} \ln(a) – \ln(b+x-1) – (r+x)\ln(\alpha + t_x) & \text{if}\ x>0 \\ 0 & \text{otherwise} \end{cases}$


```python
from scipy.special import gammaln 

def LL(params, x, t_x, T):
    if np.any(np.asarray(params) <= 0):
        return np.inf

    r, alpha, a, b = params

    ln_A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_A2 = (gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x))
    ln_A3 = -(r + x) * np.log(alpha + T)
    ln_A4 = x.copy() 
    ln_A4[ln_A4 > 0] = (np.log(a) - np.log(b + ln_A4[ln_A4 > 0] - 1) - (r + ln_A4[ln_A4 > 0]) * np.log(alpha + t_x))
    
    delta =  np.where(x>0, 1, 0)
    
    log_likelihood = ln_A1 + ln_A2 + np.log(np.exp(ln_A3) + delta * np.exp(ln_A4))
    return -log_likelihood.sum()
```

**Parameter optimization**   
이제 이 log likelihood cost fuction을 minimise하게 간단히 BFGS algorithm을 사용하면 된다.


```python
from scipy.optimize import minimize


def _func_caller(params, func_args, function):
    return function(params, *func_args)

init = np.array([1.0, 1.0, 1.0, 1.0])

sol = minimize(
    fun=_func_caller,
    method="BFGS",
    tol=0.0001, 
    x0=init,
    args=([df['x'], df['t_x'], df['T']], LL)
)

r = sol.x[0]
alpha = sol.x[1]
a = sol.x[2]
b = sol.x[3]


print(f"r = {r}")
print(f"alpha = {alpha}")
print(f"a = {a}")
print(f"b = {b}")
print(f"LL = -{sol.fun}")
```

    r = 0.24259455367664842
    alpha = 4.413602964372849
    a = 0.7929243343804369
    b = 2.4259152626264657
    LL = -9582.429206674013


### Conditional Expection

개인의 Expected sales가 앞으로 time period t앞에서 얼마나 될지 계산을 하자면  
(how many transacations any single customer will make going forward in time period t):

$$E(Y(t) \mid X=x, t_x, T, r, \alpha, a, b) = \dfrac{a + b + x – 1}{a-1} \times \dfrac{\bigg[1 – \bigg(\dfrac{\alpha + T}{\alpha + T + t}\bigg)^{r+x} {}_{2}F_{1}(r+x, b+x; a+b+x-1; \dfrac{t}{\alpha+T+t})\bigg]}{1 + \delta_{(x>0)}\dfrac{a}{b+x-1}\bigg(\dfrac{\alpha + T}{\alpha + t_x}\bigg)^{r+x}} $$


$$\text{Where } {}_{2} F_{1} \text{ is the Gaussian hypergeometric function}$$


```python
from scipy.special import hyp2f1

def calc_conditional_expectation(t, x, t_x, T):
    A = (a + b + x - 1) / (a-1)
    hyp2f1_a = r + x
    hyp2f1_b = b + x
    hyp2f1_c = a + b + x - 1
    hyp2f1_z = t / (alpha + T + t)
    hyp_term = hyp2f1(hyp2f1_a, hyp2f1_b, hyp2f1_c, hyp2f1_z)
    B = (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp_term)
    delta =  np.where(x>0, 1, 0) 
    denom = 1 + delta * (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r+x)
    return A * B / denom
```


```python
t = 39
df["conditional_expectation_upto_t_39"] = df.apply(lambda x:calc_conditional_expectation(t,df["x"],df["t_x"],df["T"])).iloc[:,-1:]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>x</th>
      <th>t_x</th>
      <th>T</th>
      <th>conditional_expectation_upto_t_39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>30.428571</td>
      <td>38.857143</td>
      <td>1.225994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1.714286</td>
      <td>38.857143</td>
      <td>0.203419</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
      <td>0.194794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
      <td>0.194794</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.000000</td>
      <td>38.857143</td>
      <td>0.194794</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2352</th>
      <td>2353</td>
      <td>0</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.258979</td>
    </tr>
    <tr>
      <th>2353</th>
      <td>2354</td>
      <td>5</td>
      <td>24.285714</td>
      <td>27.000000</td>
      <td>4.112099</td>
    </tr>
    <tr>
      <th>2354</th>
      <td>2355</td>
      <td>0</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.258979</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>2356</td>
      <td>4</td>
      <td>26.571429</td>
      <td>27.000000</td>
      <td>3.488177</td>
    </tr>
    <tr>
      <th>2356</th>
      <td>2357</td>
      <td>0</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.258979</td>
    </tr>
  </tbody>
</table>
<p>2357 rows × 5 columns</p>
</div>

