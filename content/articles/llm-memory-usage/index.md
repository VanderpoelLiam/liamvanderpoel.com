---
title: "How much memory do you need to work with LLMs?"
date: 2025-09-25T09:55:27+02:00
draft: false
---
<!-- Intro: Basics of what 1B param model means

Training from scratch (how probably dont want to do this)

Fine-tuning (quantization, llora, unsloth, etc…)

Quantization

Inference

How to ballpark numbers -->

<!-- [Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/v4.35.0/en/llm_tutorial_optimization) -->

{{< katex >}}

## What does a 1B parameter model mean?

Usually when discussing LLMs people say it is a X billion parameter model. For example the prototypical LLM [GPT-3](https://arxiv.org/pdf/2005.14165) was a 175 billion parameter model. Paremeters = model weights, all the same thing. We can treat an LLM as a collection of weight matrices, and the input prompt as a vector. As the weights are much larger than the input, we ignore it in our calculations. So these weights are floating point numbers that are usually stored as [float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format), [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), or [float16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format).

### Floating point numbers

A quick aside on floating point numbers is necessary. Consider the real number \\(\pi=3.14159265...\\). As \\(\pi\\) has infinite digits we can only store a finite number of them in memory, this is the precision of the floating point number. For a [float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) we would store \\(\pi=3.14159274101257324\\) where `3.14159274101257324 = -1^0 x 2^1 x (1 + 0.5707963705062866)` would be represented in memory as 32 bits in the form `0 10000000 10010010000111111011011` (spaces added for readability, but it would just be a single 32-bit integer). In general a decimal `(-1)^Sign x 2^(Exponent-127) x (1 + 0.Mantissa)` would be stored as a float32 in memory as the integer `Sign Exponent Mantissa` (see [Floating Point Numbers](https://www.doc.ic.ac.uk/~eedwards/compsys/float/) for more details).

The number of bits assigned to the exponent and mantissa is what differentiates a float32, bfloat16 and a float16. They each make tradeoffs between dynamic range vs precision. Precision as we have already mentioned is how many digits we support e.g. float32 has more precision than bfloat16 as the 23 bit mantissa of the float32 allows for more significant digits than the 10 bit mantissa of the bfloat16:

![Visualization of float32 compared to bfloat16 bit layouts](float32-to-bfloat16.jpeg)*float32 to bfloat16 conversion. Visualisation from [Float32 vs Float16 vs BFloat16](https://newsletter.theaiedge.io/p/float32-vs-float16-vs-bfloat16).*

Dynamic range on the other hand is how small or large of a number we can represent, this depends solely on the exponent. As the above figure shows, a bfloat16 has the same dynamic range as a float32 but takes up only 16 bits in memory vs 32 bits at the expense of lower decimal precision. On the other hand, a float16 can represent a smaller dynamic range than a bfloat16 but it has a higher precision:

![Visualization of float32 compared to float16 bit layouts](float32-to-float16.jpeg)*float32 to float16 conversion. Visualisation from [Float32 vs Float16 vs BFloat16](https://newsletter.theaiedge.io/p/float32-vs-float16-vs-bfloat16).*

The number of bits we have to represent the float is limited so we need to make tradeoff between how many bits we use in memory vs the dynamic range and precision. The motivation behind bfloat16 for LLMs is that is prevents overflow errors converting from 32-bit to 16-bit representations as the dynamic range is the same, and it only costs a little bit of precision relative to a float16.

{{< reflist exclude="wikipedia">}}
