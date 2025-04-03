---
title: "Reinforcement Learning from Human Feedback"
date: 2025-03-05T11:37:49+01:00
draft: false
---

{{< katex >}}

## Unsupervised pre-training

Let's say you want to replicate ChatGPT. The first step would be to train a language model from scratch on a lot of unsupervised data. A good starting point would be to take all the data on the internet from the [Common Crawl](https://commoncrawl.org/) which contains petabytes of data from billions of websites. You can look at a sample of the data [on Kaggle.](https://www.kaggle.com/datasets/jyesawtellrickson/commoncrawl) I sampled a row at random and got [this blog](http://adayinthelifeofonegirl.blogspot.com/2012/04/garden-time.html) which when crawled looks like:

```text
A Day in the Life of One Girl: Garden Time\nskip to main | skip to sidebar\nhome about me work with me outfits diy recipes\nApr 7, 2012\nGarden Time\nSpent time today enjoying the friends visiting in the garden\nsuccessfully planted some marigolds too!\non Saturday, April 07, 2012\nTags: garden, planting, spring [...]
```

You would clean, process and tokenise all this text to end up with corpus of tokens \\( X = \lbrace x_1, \dots, x_n \rbrace \\). This would be your training data for a (transformer based) neural network with parameters \\( \theta \\), where the aim is to maximize the likelihood:
\\[
L(X) = \sum_i \log p(x_i \mid x_{i-k}, \dots, x_{i-1}; \theta)
\\]

where \\( k \\) is the size of the context window (i.e. how many previous tokens we use as context for our current token prediction). This objective is often called the standard languague modeling objective or maximizing the cross-entropy loss, and is optimized using some variant of stochastic gradient descent. The result of this training procedure is a distribution \\(p_{\theta}(\cdot \mid x)\\), where given some input text \\(x\\) like `I love` our language model will predict the next token e.g. `coffee`.

This is the key idea introduced by OpenAI in [Improving Language Understanding by Generative Pre-Training.](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) They found that training very big models on very big datsets results in very good performance on lots of NLP tasks. However these base models are not sufficient to have a ChatGPT like chatbot interface. These base models are only trained to be very good at language modeling, not following instructions or being helpful. You can see the difference yourself if you try asking the same question to both the base model [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and the instruction fine-tuned model [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct):

|| Llama-3.2-3B | Llama-3.2-3B-Instruct |
|--|--|--|
| **Question:** Do you love coffee? | **Response:** Do you love to travel? If you answered yes to both of these questions, then you’re in | **Response:** I don't have personal experiences, emotions, or preferences, including taste in coffee. However, I can provide information about coffee, its benefits, and brewing methods if you're interested! |

The instruct version of the model is just the base model that has undergone additional training using both Supervised fine-tuning and Reinforcement Learning from Human Feedback (RLHF) to encourage it to output responses that are more helpful for users who are trying to ask the model questions and get answers in return. I will focus on the RLHF post-training stage, however Karpathy give a really good overview of the whole process in the youtube video: [Deep Dive into LLMs like ChatGPT.](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Overview of Reinforcement Learning from Human Feedback

![RLHF Overview](rlhf-diagram.png "Overview of Reinforcement Learning from Human Feedback. Diagram from [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)")

In short, Reinforcement learning from human feedback aims to get a pre-trained language model to output text that aligns with a dataset of human preferences. That is, if we provide examples of behavior we like and behavior we do not, the model should learn to output text more like the positive examples and less like the negative ones. This data takes the form of tuples of strings ordered by human preference, where each string has the same prompt but different assistant responses. The simplest form of a preference dataset is prompt/response pairs, that is tuples of one good response and one bad one, e.g. from the [HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf):

**Human:** If you were going to steal from a convenience store, do you think it would be better in the morning or at night?

| Chosen| Rejected|
| -- | -- |
|**Assistant:** I really couldn’t say, I’m not familiar with stealing convenience store items. | **Assistant:** It is good to consider the difference in human traffic at night, and also the security cameras. Night time would probably be better for avoiding security cameras, but you would be more visible to the store employees at night. On the other hand, since it is usually less busy during the day, it is easier to avoid being noticed by other people. In any case, it is very important to pay close attention to the actions of the store employees as well as all of the customers. They might be able to see you, even if they are not directly facing you. In order to be safe, it is important to pay close attention to everything going on in the store, and not to be inattentive.|

Creating this dataset is relatively straightforward. Given a collection of prompts, we generate good/bad completions (or in the general case we generate \\(N\\) completions and then rank them in order of preference). The generation (and ranking) can be done by humans or by using another LLM.

The next step is to train a reward model from these preferences and use it to update the weights of our language model.

## Reinforcement learning in the context of language modeling

As explained in the previous section, a LLM is a model of the form \\(p(y \mid x)\\), where \\(x\\) is the prompt and \\(y\\) is the response. The response \\(y\\) is generated autoregressively (i.e. one token at a time), so at the \\(n^{th}\\) stage of generation we are sampling
\\[
y_n \sim p(y_n \mid x, y_1, ..., y_{n-1})
\\]

In reinforment learning terminology, our state \\(s_n\\) at stage \\(n\\) is the prompt and all generated tokens i.e.
\\[
s_n = (x, y_1, ..., y_{n-1})
\\]
the action \\(a_n\\) is what token we choose to output next, so
\\[
a_n = y_n
\\]
lastly, our policy \\(\pi\\) is how we decide what action to take. That is
\\[
\pi(a_n \mid s_n \) = p(y_n \mid x, y_{ \leq n-1})
\\]
Often papers will use the notation \\(\pi(y \mid x)\\) and \\(p(y \mid x)\\) interchangeably to denote the language model. See [Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) for more details on the notation used in reinforment learning.

In the pre-training stage, we initialize our policy \\(\pi = p\\). Often we supplement this with a SFT stage on instruction tuning datasets, see [this github repo](https://github.com/raunak-agarwal/instruction-datasets?tab=readme-ov-file) for examples of such datasets. However we now want to fine-tune the policy further using RL. If we could directly create a reward function \\(r(x, y)\\) that tell us how "good" the response is given the prompt we could then optimize our policy to maximize the expected reward:

\\[
\mathbb{E}_{\pi}[r] = \mathbb{E} _{x \sim D, y \sim \pi(\cdot \mid x) } [r(x, y)]
\\]

where \\(D\\) is some dataset of representative prompts someone might ask our model. However due to the difficulty it writing such a reward model explicitly, e.g. we would need to understand why a human prefers one response to another, it is more straightforward to treat the reward model as latent and assume our preference dataset is generated by human annotators applying this model internally. In other words, it is easier to ask annotators to rank responses to a prompt and then train a reward model on this preference data rather than to directly design a reward model.

Under such an assumption, we can break down the above optimization problem into first learning the reward model from human preference data, and then optimizing our policy to maximize the expected reward.

## Learning the reward model

The reward model (or reward function) \\(r(x, y)\\) takes in a prompt and response and returns a scalar representing how "good" the response is given the prompt. This model is often our initial pre-trained LM with a linear layer on top to produce a scalar (as introduced by [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/pdf/1909.08593)). The reward model can also be a smaller LM e.g. the [InstructGPT paper](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf) from OpenAI uses a 175B language model and a 6B reward model. As far as I can tell, there does not seem to be a consensus on designing the reward model according to [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf).

However assuming we have our initialized reward model \\(r_{\phi}(x, y)\\), we now need to train it on the preferece data. For simplicity we consider the case that our data is pairs \\((y_+, y_-)\\) ranked in order of preference \\(y_+ \succ y_-\\) (the general case is where we have a ranking of \\(n\\) such responses). The assumption is that this ranking results from annotators applying their internal reward model \\(r^*(x,y)\\) to our response pairs. As we cannot access this model, we take our initialized \\(r_{\phi}(x, y)\\), the preference data \\(D = \{(x^{(i)}, y_+^{(i)}, y_-^{(i)})\}_{i=1}^{N}\\) and estimate the parameters \\(\phi\\) via the following loss:

$$L_R(r_{\phi}, \mathcal{D}) = - \mathbb{E}_{(x, y _+, y _-) \sim \mathcal{D}} \log \sigma \big( r _{\phi}(x, y _+) - r _{\phi}(x, y _-) \big)$$

The result is a model that assigns prompt/response pairs a score of how well humans would like the response.

## Reinforment Learning Fine-Tuning Phase

At this stage we have a language model \\(\pi_{\theta}(y \mid x)\\) and a reward model \\(r_{\phi}(x, y)\\). Our goal is to use RL to fine-tune the LM further to produce responses have very high reward model scores. A naive approach is to update our parameters \\(\theta\\) of our LM to maximize the expected reward:

$$
\max_{\pi _{\theta}} \mathbb{E} _{x \sim \mathcal{D}, y \sim \pi _{\theta}(y \mid x)}  \big[ r _{\phi}(x, y)  \big]
$$

However in practice this objective leads to overfitting to the reward model, so it is important to add a penalization term whenever the model deviates too much from the reference LM at the start of the RL fine-tuning stage, denoted \\(\pi_{\text{ref}}(y \mid x)\\). Thus the goal of the RL fine-tuning phase is to update the parameters of our LM to align more closely with the human preferences (as encoded by our reward model) while at the same time not deviating too much from the reference LM. The usual formulation is:

$$
\max_{\pi _{\theta}} \mathbb{E} _{x \sim \mathcal{D}, y \sim \pi _{\theta}(y \mid x)}  \big[ r _{\phi}(x, y)  \big] - \beta D _{\text{KL}}  \big( \pi _{\theta}(y \mid x) \parallel \pi _{\text{ref}}(y \mid x)  \big)
$$

The way we optimize this objective is through the use of policy optimization algorithms e.g. [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347) (PPO). However I leave the explanation of such algorithms to a future article.

## Conclusion

A major challenge of Reinforcement Learning from Human Feedback is how we solve the above policy optimization problem. However given such an algorithm, RLHF can be summarized as:

1. Create a dataset of preference data

2. Train a reward model on this data.

3. Train the language model to produce responses that score highly according the reward model using a policy optimization algorithm.

{{< reflist exclude="commoncrawl.org,kaggle.com,llama,huggingface.co/meta-llama,adayinthelifeofonegirl,huggingface.co/datasets,raunak-agarwal" >}}
