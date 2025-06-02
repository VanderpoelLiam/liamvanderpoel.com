---
title: "A/B Testing"
date: 2025-05-23T14:37:05+02:00
draft: false
---
{{< katex >}}

A/B testing is a way of comparing two versions of something and deciding which performs better. It is a form of hypothesis testing, where we gather evidence about how our change impacts the metrics we care about. For example, Netflix might test out a new version of their recommendation algorithm and want to decide if it increases the amount of time users spend watching movies. The classical hypothesis testing example however is the randomized control trial.

## Hypothesis Testing

Consider the following scenario:

Researchers have discovered a new miracle drug. They think it has these incredible properties, that if people take it they will live longer, be happier, etc... We want to experimentally verify if this is true, so we formulate two hypotheses:

1. Null hypothesis \\(H_0\\): The drug has no effect.

2. The alternative hypothesis \\(H_1\\): The drug has a positive effect.

We then gather some test subjects, split them into groups A and B, give group B a course of the drug and then measure the outcomes of both groups. Let us assume our outcomes are integers (where higher numbers mean a more positive outcome), we might have the following raw data:

```text
Group A: [48, 48, 48, 48, 51, 51, 47, 50, 51, 46]
Group B: [50, 51, 50, 49, 50, 49, 50, 52, 51, 51]
```

and our hypotheses are formalized in terms of the average outcome of each group:

1. Null hypothesis \\(H_0\\): \\(\mu_A = \mu_B\\).

2. The alternative hypothesis \\(H_1\\): \\(\mu_B > \mu_A\\)

Did our drug have a positive effect?

Hypothesis testing lets us answer either **Yes! There is an effect** or **We don't have any evidence of an effect**. Formally we can either:

1. **Reject the null hypothesis**: With high probability we think the drug has an effect.

2. **Fail to reject the null hypothesis**: We don't have enough evidence to claim the drug has an effect.

We cannot answer **Nope, no effect**, as it may be that we had insufficient data to detect a difference, or just got unlucky due to random chance. If we take the analogy of a murder trial. \\(H_0\\) is an innocent verdict, \\(H_1\\) is a guilty verdict. We can either find a defendant guilty (reject \\(H_0\\)) or we find them innocent (fail to reject \\(H_0\\)). But note that finding someone innocent occurs if we fail to prove guilt beyond reasonable doubt. It is not a question of "proving innocence" (accepting \\(H_0\\)) but rather failing to prove guilt.

In this experiment we have 10 test subjects in each group. Is this enough to be confident about our decision?

## Sample size

Before running an experiment we need to decide how many samples we are going to collect. Intuitively, the more samples we collect the more confident we will be about our conclusion. However we are usually constrained by a combination of time or money, and so want to pick enough samples such that we are confident enough. Therefore we need to determine the following parameters:

1. Minimum expected effect size
2. Tolerance for false positives
3. Tolerance for false negatives.

The minimum effect size is what is the smallest absolute difference in results \\(\delta\\) that we would care about. E.g. in our drug trial example, we might want to see at least a 5 year increase in lifespan, otherwise we would consider the effect too small to be worth the price of the drug.

The tolerance for false positives / negatives means what probability of a false positive or false negative are we willing to accept. The significance level \\(\alpha\\) is acceptable probability of a false positive. The power is \\(1-\beta\\) where \\(\beta\\) is the acceptable probability of a false negative. Again coming back to our drug trial example, if we have \\(\alpha = 5\\)% and \\(\beta = 20\\)% (so power of 80%) then we are willing to tolerate a 1 in 5 chance that we claim an ineffectual drug has an effect, and a 1 in 20 chance that we fail to detect that a drug improves outcomes.

I find thinking in terms of tolerance levels of adverse outcomes more helpful than thinking in terms of decision tables, but I provide them below for completeness:

| | Reject \\(H_0\\) | Fail to reject \\(H_0\\) |
|:---|:---|:---|
| \\(H_0\\) is True | False Positive | True Positive |
| \\(H_0\\) is False | True Negative | False Negative |

The same information can be expressed in terms of probabilities:

| | Probability to reject \\(H_0\\) | Probability fail to reject \\(H_0\\) |
|:---|:---|:---|
| \\(H_0\\) is True | \\(\alpha\\) | \\(1-\alpha\\) |
| \\(H_0\\) is False | \\(1-\beta\\) | \\(\beta\\) |

Lets assume we have picked all three of these parameters. The last piece of information we need is whether the alternative hypothesis is directional. For example, `the drug has a positive effect` is directional while `the drug has an effect` is not. This matters as we will need to adjust our significance level accordingly, as there are twice as many ways to get a false positive for a non-directional null-hypothesis than a directional one. Hence to maintain our desired false positive rate of \\(\alpha\\), we use the stricter rate of \\(\alpha / 2\\) in our sample size calculation below.

Given all this information, the paper [So you want to run an experiment, now what? Some Simple Rules of Thumb for Optimal Experimental Design.](https://www.nber.org/system/files/working_papers/w15701/w15701.pdf) explains how to pick the number of samples \\(n\\) per group in section 3.1:

$$
n = (z_{\alpha} + t_{\beta})^2 \cdot (\sigma_A^2 + \sigma_B^2) \cdot \frac{1}{\delta^{2}}
$$

where \\(\sigma_A\\), \\(\sigma_B\\) are the standard deviations of groups A and B, and \\(z_{\alpha}\\) is the z-score of \\(\alpha\\) i.e. the solution to the equation

$$
P(Z > z_{\alpha}) = \alpha
$$

where \\(Z\\) is the standard normal. In words \\(z_{\alpha}\\) is the point where we expect a \\(\alpha\\) chance of a data point drawn from \\(Z\\) being larger than this value. In python this can be implemented as:

```python
from scipy.stats import norm

alpha = 0.05 # 5%
z_alpha = norm.ppf(1 - alpha)
```

This formula assumes we have the same number of samples \\(n\\) per group, however we often have that the treatment group B is smaller than the control group A. We therefore would also need to pick the ratio of samples per group, let \\(\pi_A\\) denote the ratio of samples assigned to group A (such that \\(\pi_A + \pi_B = 1\\)), then our formula for the total number of required samples becomes:

$$
n = (z_{\alpha} + t_{\beta})^2 \cdot \left( \frac{\sigma_A^2}{\pi_A} + \frac{\sigma_B^2}{\pi_B} \right) \cdot \frac{1}{\delta^{2}}
$$

which we would allocate according to our ratios \\(\pi_A\\) and \\(\pi_B\\).

Lastly, if we have not yet run our experiment how can we know the standard deviations \\(\sigma_A\\), \\(\sigma_B\\)? Well we would either need to estimate it empirically e.g. \\(\bar{\mu} = \frac{1}{N} \sum x_i\\) and \\(\bar{\sigma} = \frac{1}{N} \sum (\bar{\mu} - x_i)^2\\) based on some baseline results and extrapolate to the treatment group. Or if we are dealing with [Bernoulli random variables](https://en.wikipedia.org/wiki/Bernoulli_distribution) we know  \\(\sigma^2 = \mu \cdot (1-\mu)\\), so we only need to estimate the baseline mean \\(\mu_A\\) and can then set the treatment mean to \\(\mu_B = \mu_A + \delta\\).

## p-values

TODO: What if I just eyeballed the sample size and picked n = 16\frac{\sigma^2}{\delta^2}, what is my confidence in my result?

TODO: Next steps are p-values and how to avoid invalidating statistical significance by stopping experiments early or peeking at results.

<!-- ### Selecting the sample size

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

Good news right? At this point I should mention that I generated the data by sampling 20 times from \\(\mathcal{N}(50, 2)\\) and then splitting the data in half. So if the samples are all drawn from the same distribution why are we getting a statistically significant result?

TODO: In practice not needed to derive everything from scratch, can use online calculator like [Evan Miller Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html) or [GoDaddy maintained Python package](https://github.com/godaddy/sample-size). -->

<!-- TODO: Explain with formula from here: https://www.nber.org/system/files/working_papers/w15701/w15701.pdf what the smallest difference we can measure is? Also explain that CLT has assumptions on n being sufficiently large that are likely not met -->

<!-- TODO: Makes sense to layout article more like in https://bytepawn.com/ab-testing-and-the-ztest.html -->

{{< reflist >}}

<!-- Relevant links:

- [Monte Carlo Power Analysis](https://deliveroo.engineering/2018/12/07/monte-carlo-power-analysis.html)
- [The Unreasonable Effectiveness of Monte Carlo Simulations in A/B Testing](https://bytepawn.com/unreasonable-effectiveness-monte-carlo-ab-testing.html)
- [Building intuition for p-values and statistical significance](https://bytepawn.com/building-intuition-p-values-statistical-significance.html#building-intuition-p-values-statistical-significance)
- [Beautiful A/B testing](https://bytepawn.com/beautiful-ab-testing.html#beautiful-ab-testing) -->
