---
title: "A/B Testing"
date: 2025-05-23T14:37:05+02:00
draft: false
---
{{< katex >}}

## Hypothesis Testing

Consider the following scenario:

Researchers have discovered a new miracle drug. They think it has these incredible properties, that if people take it they will live longer, be happier, etc... We want to experimentally verify if this is true, so we formulate two hypotheses:

1. Null hypothesis \\(H_0\\): The drug has no effect.

2. The alternative hypothesis \\(H_1\\): The drug has an effect.

We then gather some test subjects, split them into treatment/control groups, give them a course of the drug and then measure the outcomes. Let us assume our outcomes are integers, we might have the following raw data:

```text
Treatment group: [48, 48, 48, 48, 51, 51, 47, 50, 51, 46]
Control group: [50, 51, 50, 49, 50, 49, 50, 52, 51, 51]
```

Did our drug have an effect?

Hypothesis testing lets us answer either **Yes! There is an effect** or **We don't have any evidence of an effect**. Formally we can either:

1. **Reject the null hypothesis**: With high probability we think the drug has an effect.

2. **Fail to reject the null hypothesis**: We don't have enough evidence to claim the drug has an effect.

We cannot answer **Nope, no effect**, as it may be that we had insufficient data to detect a difference, or just got unlucky due to random chance. If we take the analogy of a murder trial. \\(H_0\\) is an innocent verdict, \\(H_1\\) is a guilty verdict. We can either find a defendant guilty (reject \\(H_0\\)) or we find them innocent (fail to reject \\(H_0\\)). But note that finding someone innocent occurs if we fail to prove guilt beyond reasonable doubt. It is not a question of "proving innocence" (accepting \\(H_0\\)) but rather failing to prove guilt.

Therefore, an important part of hypothesis testing is picking appropriate hyperparameters such that the probability of detecting an effect is within our tolerance levels.

### Power Analysis

The hyperparameters we need to decide on before running our experiment are:

1. Minimum effect size
2. Tolerance for false positives
3. Tolerance for false negatives.

The minimum effect size (change in our statistic) we want to be able to detect is a bit of a judgement call, but usually we have some intuition about the expected response, e.g. in our drug example I would expect some people not to respond even if the drug works. Next, in this context false positives/negatives mean the following:

| | Reject \\(H_0\\) | Fail to reject \\(H_0\\) |
|:---|:---|:---|
| \\(H_0\\) is True | False Positive | True Positive |
| \\(H_0\\) is False | True Negative | False Negative |

Our tolerance level is what probability of a false positive or false negative are we willing to accept. The significance level \\(\alpha\\) is the probability of a false positive we are willing to accept. The power is \\(1-\beta\\) where \\(\beta\\) is the probability of a false positive we are willing to accept. We can therefore represent the above table as:

| | Probability to reject \\(H_0\\) | Probability fail to reject \\(H_0\\) |
|:---|:---|:---|
| \\(H_0\\) is True | \\(\alpha\\) | \\(1-\alpha\\) |
| \\(H_0\\) is False | \\(1-\beta\\) | \\(\beta\\) |

### Selecting the sample size

So we now have the following hyperparameters set for our experiment:

1. Minimum effect size: \\(\delta\\)
2. Tolerance for false positives: \\(\alpha\\)
3. Tolerance for false negatives: \\(\beta\\)

We are now ready to run our experiment, but how many samples should we collect? The article by Evan Miller [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) explains that in order to avoid "repeated significance testing errors" we need to fix the sample size in advance. Have a look at the article for an explanation for why this is so important. To fix the sample size in advance, a good rule of thumb is:

$$
n = 16\frac{\sigma^2}{\delta^2}
$$

Where \\(\sigma^2\\) is the expected sample variance. We likely do not know the variance in advance, but it can be estimated through Monte Carlo methods (which we will come to later) or if we are dealing with a statistic sampled from a binomial distribution (e.g. our data is a series of binary outcomes and we compute the success rate) we often can estimate the success probability \\(p\\), then the variance is just \\(\sigma^2 = p \cdot (1-p)\\).

The paper [So you want to run an experiment, now what? Some Simple Rules of Thumb for Optimal Experimental Design.](https://www.nber.org/system/files/working_papers/w15701/w15701.pdf) provides a better estimate of \\(n\\) in section 3.1 given our specific choice of \\(\alpha\\) and \\(\beta\\):

$$
n = 2 \cdot (t_{\alpha/2} + t_{\beta})^2 \cdot \frac{\sigma^2}{\delta^2}
$$

where \\(t_{\alpha}\\) is the solution to the equation

$$
P(Z > t_{\alpha}) = \alpha
$$

Where \\(Z\\) is the standard normal. In python this would look like:

```python
from scipy.stats import norm

alpha = 0.05
t_alpha = norm.ppf(1 - alpha)
```

Another useful trick is to specify the minium effect size as a fraction of the standard deviation. For example, if we want to be able to detect a 0.2 standard deviation change we set \\(\delta = 0.2 \sigma \\), and we select significance \\(\alpha = 0.01\\) and power \\(1-\beta = 0.9\\), the number of samples we need are:

```python
import math
from scipy.stats import norm

alpha = 0.01
power = 0.9
min_effect_ratio = 0.2
t_alpha = norm.ppf(1 - alpha)
t_beta = norm.ppf(power)


n = math.ceil(2 * (t_alpha + t_beta)**2 * min_effect_ratio**(-2))
print(n) # 651
```

#### Size of effect we can detect

If we are running our experiment, we should not be reporting significance levels until the experiment is over. However we can still report the size of the effect we can detect given the current sample size:

$$
\delta = \sigma \cdot (t_{\alpha/2} + t_{\beta}) \cdot \sqrt{\frac{2}{n}}
$$

### p-values and statistical significance

Lets say we ran the experiment having selecting our tolerance for error, our expected effect size and plugged this into the above formula to pick the number of samples. We got some data that looks like this:

```text
Group A: [48, 48, 48, 48, 51, 51, 47, 50, 51, 46]
Group B: [50, 51, 50, 49, 50, 49, 50, 52, 51, 51]
```

We now come back the the question: Did our drug have an effect?

To perform a hypothesis test, we need to do two things:

1. Model the distribution of \\(H_0\\). This is our test statistic.

2. Compute how likely it is that our samples were generated under \\(H_0\\). This is our p-value.

Define null hypothesis more precisely as the means are the same for both groups, and the alternative hypothesis that they are not the same. According to the Central Limit Theorem (CLT) the sample distribution of the mean converges to a normal distribution as \\(n \to \infty\\). So under the null hypothesis, we can model:

$$
\frac{\mu_B - \mu_A}{\sqrt{\frac{\sigma_B^2 + \sigma_A^2}{n}}} \sim \mathcal{N}(0, 1)
$$

where \\(\mu_A, \mu_B\\) and and \\(\sigma_A^2, \sigma_B^2\\) are the true means and variances respectively. See [The art of A/B testing](https://archive.ph/8Bp8p) for more details on the derivation. We then compute a test statistic \\(\hat{z}\\) from the empirical means and variances:

$$
\hat{z} = \frac{\hat{\mu}_B - \hat{\mu}_A}{\sqrt{\frac{\hat{\sigma}_B^2 + \hat{\sigma}_A^2}{n}}}
$$

Under the null hypothesis, this is a sample from a standard normal. The p-value is a measure of how surprising our result is under the null hypothesis. A low p-value means our result is very unexpected, and suggests that our null hypothesis may not be correct. In general, the p-value is:

$$
\text{p-value} = P(\text{Test statistic is as or more extreme than observed} \mid H_0)
$$

for our given test statistic and alternate hypothesis this is:

$$
\text{p-value} = P(|Z| \geq |\hat{z}|)
$$

Note that if our alternative hypothesis was instead \\(\mu_B > \mu_A\\), the p-value would instead be \\(P(Z \geq \hat{z})\\) as the alternative hypothesis is one-sided. However given the stated alternative hypothesis, putting everything together we would compute the p-value in Python as follows (adapted from [A/B testing and the Z-test](https://bytepawn.com/ab-testing-and-the-ztest.html)):

```python
import numpy as np
from scipy.stats import norm


def z_to_p(z, one_sided):
    p = 1 - norm.cdf(z)
    if one_sided:
        return p
    else:
        return 2*p


A = np.array([48, 48, 48, 48, 51, 51, 47, 50, 51, 46])
B = np.array([50, 51, 50, 49, 50, 49, 50, 52, 51, 51])
N = 10

alpha = 0.05


mu_A = np.mean(A)
mu_B = np.mean(B)

var_A = np.var(A)
var_B = np.var(B)

z_hat = (mu_B - mu_A) / np.sqrt((var_A + var_B) / N)

p = z_to_p(z_hat, one_sided=False)

print('Empirical mean for A: %.3f' % mu_A)
print('Empirical mean for B: %.3f' % mu_B)

print('p-value: %.3f' % p)
if p <= alpha:
    print("""The miracle drug works!""")
else:
    print("""We're not sure if the drug works.""")
```

This outputs:

```text
Empirical mean for A: 48.800
Empirical mean for B: 50.300
p-value: 0.015
The miracle drug works!
```

Good news right? At this point I should mention that I generated the data by sampling 20 times from \\(\mathcal{N}(50, 2)\\) and then splitting the data in half. So if the samples are all drawn from the same distribution why are we getting a statistically significant result.

<!-- TODO: Explain with formula from here: https://www.nber.org/system/files/working_papers/w15701/w15701.pdf what the smallest difference we can measure is? Also explain that CLT has assumptions on n being sufficiently large that are likely not met -->

<!-- TODO: Makes sense to layout article more like in https://bytepawn.com/ab-testing-and-the-ztest.html -->


{{< reflist >}}

Relevant links:

- [Monte Carlo Power Analysis](https://deliveroo.engineering/2018/12/07/monte-carlo-power-analysis.html)
- [The Unreasonable Effectiveness of Monte Carlo Simulations in A/B Testing](https://bytepawn.com/unreasonable-effectiveness-monte-carlo-ab-testing.html)
- [Building intuition for p-values and statistical significance](https://bytepawn.com/building-intuition-p-values-statistical-significance.html#building-intuition-p-values-statistical-significance)
- [Beautiful A/B testing](https://bytepawn.com/beautiful-ab-testing.html#beautiful-ab-testing)
