---
title: "Monte Carlo Simulations and A/B Testing"
date: 2025-05-23T14:37:05+02:00
draft: false
---
{{< katex >}}

## Hypothesis Testing

Consider the following scenario:

Researchers have discovered a new miracle drug. They think it has these incredible properties, that if people take it they will live longer, be happier, etc... We want to experimentally verify if this is true, so we formulate two hypotheses:

1. Null hypothesis \\(H_0\\): The drug has no effect.

2. The alternative hypothesis \\(H_1\\): The drug has an effect.

<!-- TODO: This isn't quite right, raw data would be what I showed and test statistic would be the mean. -->

We then gather some test subjects, split them into treatment/control groups, give them a course of the drug and then measure the outcomes. We need to then condense these outcomes into a [Test Statistic](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/hypothesis-testing/test-statistic.html), essentially a numerical summary of our data. Lets assume our test statistic is a single integer, so our results might look like:

```text
Treatment group: [48, 48, 48, 48, 51, 51, 47, 50, 51, 46]
Control group: [50, 51, 50, 49, 50, 49, 50, 52, 51, 51]
```

Did our drug have an effect?

Hypothesis testing lets us answer either **Yes! There is an effect** or **We don't have any evidence of an effect**. Formally we can either:

1. **Reject the null hypothesis**: With high probability we think the drug has an effect.

2. **Fail to reject the null hypothesis**: We don't have enough evidence to claim the drug has an effect.

We cannot answer **Nope, no effect**, as it may be that we had insufficient data to detect a difference, or just got unlucky due to random chance.

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


<!-- TODO: How does this calculator work? https://www.evanmiller.org/ab-testing/sample-size.html
TODO: How to pick the sample size: START HERE - https://www.evanmiller.org/how-not-to-run-an-ab-test.html -->


{{< reflist >}}

Relevant links:

- [Monte Carlo Power Analysis](https://deliveroo.engineering/2018/12/07/monte-carlo-power-analysis.html)
- [The Unreasonable Effectiveness of Monte Carlo Simulations in A/B Testing](https://bytepawn.com/unreasonable-effectiveness-monte-carlo-ab-testing.html)
- [Building intuition for p-values and statistical significance](https://bytepawn.com/building-intuition-p-values-statistical-significance.html#building-intuition-p-values-statistical-significance)
- [Beautiful A/B testing](https://bytepawn.com/beautiful-ab-testing.html#beautiful-ab-testing)
