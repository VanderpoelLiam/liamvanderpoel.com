---
title: "A/B Testing"
date: 2025-05-23T14:37:05+02:00
draft: false
---
{{< katex >}}

A/B testing is a way of comparing two versions of something and deciding which performs better. It is a form of hypothesis testing, where we gather evidence about how our change impacts some metrics we care about and then based on this evidence make a decision whether to use the new version or keep the old one. For example, Netflix might test out a new variation of their recommendation algorithm and want to know if it increases the amount of time users spend watching movies. The classical hypothesis testing example however is the randomized control trial.

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

## Sample Size Selection

### How Many Samples is Enough?

Before running our drug trial experiment we need to decide how many test subject to use. Intuitively, the more subjects we use the more confident we will be about our conclusion. However we are usually constrained by a combination of time or money, and so want to pick enough samples to be "confident enough". Therefore we need to determine the following parameters:

1. Minimum expected effect size
2. Tolerance for false positives
3. Tolerance for false negatives.

The minimum effect size is what is the smallest absolute difference in results \\(\delta\\) that we would care about. E.g. in our drug trial example, we might want to see at least a 5 year increase in lifespan, otherwise we would consider the effect too small to be worth the price of the drug.

The tolerance for false positives / negatives means what probability of a false positive or false negative are we willing to accept. The significance level \\(\alpha\\) is acceptable probability of a false positive. The power is \\(1-\beta\\) where \\(\beta\\) is the acceptable probability of a false negative. Again coming back to our drug trial example, if we have \\(\alpha = 5\\)% and \\(\beta = 20\\)% (so power of 80%) then we are willing to tolerate a 1 in 20 chance that we claim an ineffectual drug has an effect, and a 1 in 5 chance that we fail to detect that a drug improves outcomes.

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

Lets assume we have picked all three of these parameters. The last piece of information we need is whether the alternative hypothesis is directional. For example, `the drug has a positive effect` is directional while `the drug has an effect` is not. This matters as we will need to adjust our significance level accordingly, as there are twice as many ways to get a false positive for a non-directional alternative hypothesis than a directional one. Hence to maintain our desired false positive rate of \\(\alpha\\), we would use the stricter significance level of \\(\alpha / 2\\) in our sample size calculation below to account for a non-directional alternative hypothesis (also called a two-sided outcome).

### Sample Size Formula

Given all this information, the paper [So you want to run an experiment, now what? Some Simple Rules of Thumb for Optimal Experimental Design](https://www.nber.org/system/files/working_papers/w15701/w15701.pdf) derives a formula for how to pick the total number of samples \\(n\\) in section 3.1:

$$
n = 2 \cdot (z_{\alpha} + z_{\beta})^2 \cdot (\sigma_A^2 + \sigma_B^2) \cdot \frac{1}{\delta^{2}}
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

This formula assumes we want to have the same number of samples per group, however we often want the treatment group B to be smaller than the control group A (as a hedge in case the treatment is much worse). Let \\(\pi_A\\) denote the ratio of samples assigned to group A (such that \\(\pi_A + \pi_B = 1\\)), then our formula for the total number of required samples becomes:

$$
n = (z_{\alpha} + z_{\beta})^2 \cdot \left( \frac{\sigma_A^2}{\pi_A} + \frac{\sigma_B^2}{\pi_B} \right) \cdot \frac{1}{\delta^{2}}
$$

which we would allocate according to our ratios \\(\pi_A\\) and \\(\pi_B\\). This simplifies to the previous formula for \\(\pi_A = \pi_B = 0.5\\).

Lastly, if we have not yet run our experiment how can we know the standard deviations \\(\sigma_A\\), \\(\sigma_B\\)? We would either need to estimate it empirically e.g. calculate \\(\hat{\mu} = \frac{1}{N} \sum x_i\\) and \\(\hat{\sigma} = \frac{1}{N} \sum (\hat{\mu} - x_i)^2\\) based on some baseline results, and assume the value is close enough to the real standard deviation. Or if we are dealing with [Bernoulli random variables](https://en.wikipedia.org/wiki/Bernoulli_distribution) we know  \\(\sigma^2 = \mu \cdot (1-\mu)\\), so a trick is to estimate the baseline mean \\(\mu_A\\) then set the treatment mean to \\(\mu_B = \mu_A + \delta\\). Adapting code from [A/B testing and the Z-test](https://bytepawn.com/ab-testing-and-the-ztest.html) we can calculate the necessary sample size in Python:

```python
import math
from scipy.stats import norm

def minimum_num_samples(var_A, var_B, pi_A, delta, alpha, power, one_sided):
    """Calculate the minimum total number of samples to achieve desired confidence levels.

    Args:
        var_A (float): The baseline variance.
        var_B (float): The treatment variance.
        pi_A (float): The ratio of samples assigned to the baseline.
        delta (float): The absolute minimum detectable effect size.
        alpha (float): The significance level.
        power (float): The power of the test.
        one_sided (bool): Whether the alternative hypothesis is directional.
    """
    if one_sided:
        z_alpha = norm.ppf(1 - alpha)
    else:
        z_alpha = norm.ppf(1 - alpha / 2)
  
    z_power  = norm.ppf(power)
    pi_B = 1 - pi_A 
    N = ((z_alpha + z_power) ** 2) * (var_A/pi_A + var_B/pi_B) / (delta) ** 2 
    return math.ceil(N)
```

In practice you do not need to write this code from scratch as you can either use an online calculator like [Evan Miller Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html) or an open source solution like [GoDaddy sample-size Python package](https://github.com/godaddy/sample-size).

## p-values and Statistical Significance

### Computing the p-value

Recall, we ran our experiment with 20 samples and got some data that looks like this:

```text
Group A: [48, 48, 48, 48, 51, 51, 47, 50, 51, 46]
Group B: [50, 51, 50, 49, 50, 49, 50, 52, 51, 51]
```

To perform a hypothesis test and answer the question `Did our drug have a positive effect?`, we need to do two things:

1. Choose a test statistic that summarizes our data as a single number, and then model the distribution of the test statistic under the assumption that \\(H_0\\) is true.

2. Compute how likely it is that our samples were generated under this model. This is our p-value.

Recall that our null hypothesis is the means are the same for both groups (\\(\mu_A = \mu_B\\)), and the alternative hypothesis that the mean of group B is larger (\\(\mu_B > \mu_A\\)).

According to the Central Limit Theorem (CLT) the sample distribution of the mean converges to a normal distribution as \\(n \to \infty\\). So per the CLT if \\(n\\) is sufficiently large, we can model the difference in sample means as the following normal distribution:

$$
\hat{\mu}_B - \hat{\mu}_A \sim \mathcal{N}(\mu_B - \mu_A, \frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B})
$$

where \\(\mu_A, \mu_B\\) and \\(\sigma_A^2, \sigma_B^2\\) are the true means and variances and \\(\hat{\mu}_A,\hat{\mu}_B\\) are the sample means. Under the null hypothesis \\(\mu_B - \mu_A = 0\\), so our distribution becomes:

$$
\hat{\mu}_B - \hat{\mu}_A \sim \mathcal{N}(0, \frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B})
$$

and as the true variances are not known, we estimate them empirically:
$$
\hat{\mu}_B - \hat{\mu}_A \sim \mathcal{N}(0, \frac{\hat{\sigma}_A^2}{n_A} + \frac{\hat{\sigma}_B^2}{n_B})
$$

The last step is to decide on a test statistic, we could just pick the left hand side from the above equation as is, but it is common to first standardize to:
$$
\frac{\mu_B - \mu_A}{\sqrt{\frac{\hat{\sigma}_A^2}{n_A} + \frac{\hat{\sigma}_B^2}{n_B}}} \sim \mathcal{N}(0, 1)
$$

We now have a model of our test statistic under the null hypothesis, and the value of the test statistic \\(\hat{z}\\) according to our data is given by plugging the empirical means into the above equation to get:

$$
\hat{z} = \frac{\hat{\mu}_B - \hat{\mu}_A}{\sqrt{\frac{\hat{\sigma}_A^2}{n_A} + \frac{\hat{\sigma}_B^2}{n_B}}}
$$

This completes the first step, see [The art of A/B testing](https://archive.ph/8Bp8p) for more details on the derivation. Step two is to determine the p-value of the test statistic. The p-value is a measure of how surprising our result is under the null hypothesis. A low p-value means our result is very unexpected, and suggests that our null hypothesis is not correct. In general, the p-value is:

$$
\text{p-value} = P(\text{Test statistic is as or more extreme than observed} \mid H_0)
$$

for our given test statistic and alternate hypothesis this is:

$$
\text{p-value} = P(Z \geq \hat{z})
$$

Note that if our alternative hypothesis was instead two-sided, as in \\(\mu_B \neq \mu_A\\), the p-value would instead be \\(P(|Z| \geq |\hat{z}|)\\).

We can compute the p-value in Python as follows (adapted from [A/B testing and the Z-test](https://bytepawn.com/ab-testing-and-the-ztest.html)):

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
z_hat = (B.mean() - A.mean()) / np.sqrt((A.var() + B.var()) / N)

p = z_to_p(z_hat, one_sided=True)

print('Empirical mean for A: %.3f' % mu_A)
print('Empirical mean for B: %.3f' % mu_B)

print('p-value: %.3f' % p)
if p <= alpha:
    print("""The miracle drug works!""")
else:
    print("""We're not sure if the drug works.""")

>>> "Empirical mean for A: 48.800"
>>> "Empirical mean for B: 50.300"
>>> "p-value: 0.007"
>>> "The miracle drug works!"
```

Good news right? The p-value is less than our significance level, so we have a statistically significant difference right?

Well ... the way I created this data was by sampling 20 times from \\(\mathcal{N}(50, 2)\\), splitting the data in half and then setting group B to the split with larger mean. Therefore this "statistically significant" difference is completely due to random chance.

### Importance of Sample Size

Using the default confidence levels, if we are looking for a 1% change in the mean, then plugging our parameters into our sample size formula:

```python
alpha = 0.05
power = 0.8
min_delta = 1/100 * mu_A
n_samples = minimum_num_samples(var_A, var_A, 0.5, min_delta, alpha, power, one_sided=True)
print(f"Minimum number of samples: {n_samples}")

>>> "Minimum number of samples: 308"
```

We should have used 308 samples to get our desired confidence levels, instead we used 20, what does this mean for our conclusions? First, with limited sample size many of the assumptions we make to run the hypothesis test break down, for example the CLT probably doesn't apply, and so our model of the test statistic is not justified. Therefore any conclusions of significant results based on the computed p-values are meaningless. Second, with small sample sizes the false positive rate of our experiment is much higher. In general we cannot calculate the false positive rate of an hypothesis test given a sample size. But if you make an assumption that you can model the underlying process sufficiently well, then we can just simulate the test many times and estimate how often we get a false positive. This approach is called Monte Carlo simulation, and based on the code from [The Unreasonable Effectiveness of Monte Carlo Simulations in A/B Testing](https://bytepawn.com/unreasonable-effectiveness-monte-carlo-ab-testing.html) and that our data is drawn from \\(\mathcal{N}(50, 2)\\) we can estimate the false positive rate for `N=20` and `N=308` as follows:

```python
def simulate_ab_test(N, lift):
    """Simulate our drug trial AB test and return the p-value."""
    mu, sigma = 50, 2
    A = np.random.normal(mu, sigma, N)
    B = np.random.normal(mu + lift, sigma, N)
    z_hat = (B.mean() - A.mean()) / np.sqrt((A.var() + B.var()) / N)
    p = z_to_p(z_hat, one_sided=True)
    return p
    
alpha = 0.05
N = 10
lift = 0
num_simulations = 100000

p_values = [simulate_ab_test(N, lift) for _ in range(num_simulations)]
fp_rate = np.sum(np.array(p_values) < alpha) / num_simulations
print(f"Estimated false positive rate: {fp_rate:.2%} for {N} samples")

N = 308
p_values = [simulate_ab_test(N, lift) for _ in range(num_simulations)]
fp_rate = np.sum(np.array(p_values) < alpha) / num_simulations
print(f"Estimated false positive rate: {fp_rate:.2%} for {N} samples")

>>> "Estimated false positive rate: 6.74% for 10 samples"
>>> "Estimated false positive rate: 4.93% for 308 samples"
```

The simulation shows that the false positive rate is `37%` greater when we take 20 samples instead of 308. It also shows that even when there is no true difference, we expect to see some false positives due to chance. 

## Avoiding Statistical Sins



TODO: Rework this entire section

<!-- Monte Carlo simulations can also provide insight into another big problem when running A/B tests which is early stopping.

### Early Stopping

How do we now proceed? Do we throw more test subjects at the experiment until we have 74 samples? The problem is something called `repeated significance testing errors`. Repeatedly checking the results by running our statistical test multiple times causes the false positive rate to skyrocket. Have a look at [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html) for a more detailed explanation on why this occurs, but in short the best way to avoid this issue is to not repeatedly test for significance. We should fix the sample size in advance before running the experiment, and not report any significance results until the experiment is over. We should especially not use the a significant result to stop the test, else you might get a false positive like we achieved earlier.

The three articles I draw from in this section are [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html), [Simple Sequential A/B Testing](https://www.evanmiller.org/sequential-ab-testing.html) and [A/B Testing Rigorously (without losing your job)](https://elem.com/~btilly/ab-testing-multiple-looks/part1-rigorous.html).

TODO: Explain problem of peeking at the data and point to [Simple Sequential A/B Testing](https://www.evanmiller.org/sequential-ab-testing.html)

TODO: How to avoid invalidating statistical significance by stopping experiments early or peeking at results. [How Not To Run an A/B Test](https://www.evanmiller.org/how-not-to-run-an-ab-test.html).

TODO: Explain connection to sequential analysis, but that beyond scope this article. But in short, probably fine to continue adding test subject 

### Minimum Effect Size

If we go back to the paper [So you want to run an experiment, now what? Some Simple Rules of Thumb for Optimal Experimental Design](https://www.nber.org/system/files/working_papers/w15701/w15701.pdf) we can determine how large of an effect can be detected given the current sample size:

$$
\delta = (z_{\alpha} + z_{\beta}) \cdot \sqrt{\frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B}}
$$

In Python this looks like:

```python
from scipy.stats import norm

def minimum_detectable_effect_size(var_A, var_B, n_A, n_B, alpha, power, one_sided):
    """Calculate the minimum detectable effect size.

    Returns the absolute difference between the two means that is detectable
    with the given parameters.

    Args:
        var_A (float): The baseline variance.
        var_B (float): The treatment variance. 
        n_A (int): The number of samples in the first experiment.
        n_B (int): The number of samples in the second experiment.
        alpha (float): The false positive rate we are willing to accept.
        power (float): (1-power) is the false negative rate we are willing to accept.
        one_sided (bool): Whether the alternative hypothesis is directional.

    Returns:
        float: The minimum detectable effect size.
    """
    if one_sided:
        z_alpha = norm.ppf(1 - alpha)
    else:
        z_alpha = norm.ppf(1 - alpha / 2)

    z_power = norm.ppf(power)
    

    delta = (z_alpha + z_power) * math.sqrt(var_A / n_A + var_B / n_B)
    return delta
```

and if we run this function on our data:

```python
alpha = 0.05
power = 0.8

min_delta = minimum_detectable_effect_size(var_A, var_B, N, N, alpha, power, one_sided=True) 

print(f"Minimum detectable effect size: {min_delta:.2f}")
print(f"Actual effect size: {mu_B - mu_A:.2f}")

>>> "Minimum detectable effect size: 1.53"
>>> "Actual effect size: 1.50"
``` -->

{{< reflist >}}

<!-- Relevant links:

- [Monte Carlo Power Analysis](https://deliveroo.engineering/2018/12/07/monte-carlo-power-analysis.html)
- [The Unreasonable Effectiveness of Monte Carlo Simulations in A/B Testing](https://bytepawn.com/unreasonable-effectiveness-monte-carlo-ab-testing.html)
- [Building intuition for p-values and statistical significance](https://bytepawn.com/building-intuition-p-values-statistical-significance.html#building-intuition-p-values-statistical-significance)
- [Beautiful A/B testing](https://bytepawn.com/beautiful-ab-testing.html#beautiful-ab-testing) -->
