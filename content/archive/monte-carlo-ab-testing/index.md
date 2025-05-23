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

Therefore, an important part of hypothesis testing is picking appropriate hyperparameters to set our tolerance levels for what level of certainly we want from our experiment.

### Statistical Power





<!-- Our data is a sequence of test statistics sampled from two distributions. So we can represent our data as two sequences: \\(x_1, x_2, ..., x_n \sim F_1\\) and \\(y_1, y_2, ..., y_m \sim F_2\\). Due to the randomness inherent in sampling, it is possible that even if the distributions are the same we end up with very different samples.  -->


<!-- Well I created this data by sampling from a normal distribution with mean 50 and variance 2, so the answer should be no.  -->


<!-- TODO: Experimental Design continuation. Think makes sense to start with experimental design, hypothesis testing, then statistical power, then MC methods.

## Monte Carlo Methods

TODO: Explain motivation behind their use.

## A/B Testing

TODO: Explain use cases of A/B testing

### Experimental Design

TODO: Experimental design given 1. we want to detect a minimum effect size 2. what sample size do we need to have a good chance of detecting this minimum effect

## Hypothesis Testing and Types of Errors

TODO: What is hypothesis testing? False positive vs False negative.

### Statistical Power

The minimum effect size we want to be able to detect is a hyperparameter that needs to be selected prior to starting an experiment. This is often a judgement call, and is required in order to answer the question: What sample size do we need to have a good chance of detecting the minimum effect? -->

{{< reflist >}}
