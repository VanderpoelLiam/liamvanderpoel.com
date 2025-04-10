---
title: "Data Structures and Algorithms"
date: 2025-04-08T11:05:46+02:00
draft: false
---
{{< katex >}}

## Data Structures

<!-- 1. Knowing the time and space complexity of each.

2. Knowing how to implement them.

3. Most importantly, knowing how to use them in your programming language of choice. -->

Reference for [time-complexity of list, queue and set in python](https://wiki.python.org/moin/TimeComplexity).

### range()

The function has the form `range(start, stop, step)`, see [docs](https://docs.python.org/2/library/functions.html#range). We go from start with increment step until the next step would exceed (or be less than for negative step) stop. Examples:

```python
>> list(range(0, 10, 2))
[0, 2, 4, 6, 8]
>>> list(range(0, 10, 6))
[0, 6]
>>> list(range(5, 0, -1))
[5, 4, 3, 2, 1]
```

### String slicing

Similar to range for a string `s` we have `s[start:stop]` that goes includes the start element but not the stop element. Examples:

```python
>> s = "012345"
>> s[0:2]
"01"
>> s[1:4]
"123"
>> s[3:]
"345"
>> s[:5]
"01234"
```

Can optionally include a `step` i.e. `s[start:stop:step]` which increments the index by step each time. Examples:

```python
>> s = "012345"
>> s[1:4:2]
"13"
>> s[::2]
"024"
```

### Dynamic Arrays

Implememted as `list` in Python.

| Operation | Python | Time Complexity |
|-----------|---------|-----------------|
| Create empty list | `lst = []` | O(1) |
| Create list with items | `lst = [1,2,3]` | O(n) |
| Append item | `lst.append('a')` | O(1) |
| Delete by index | `del lst[0]` | O(n) |
| Remove first occurrence | `lst.remove(x)` | O(n) |
| Check if element exists | `x in lst` | O(n) |
| Pop last item | `lst.pop()` | O(1) |
| Pop item at index | `lst.pop(i)` | O(n) |

### Linked Lists

TODO

### Stacks

TODO

### Queues

TODO

### Binary Trees

TODO

### Hash Maps / Sets

TODO

### Heaps

TODO

### Graphs

TODO

## Algorithms

### Bucket Sort

This algorithm is useful for problems where we want to get the \\(k\\) most frequent elements in a list.

**Examples**

```text
Input: nums = [1,2,2,3,3,3], k = 2

Output: [2,3]
```

```text
Input: nums = [7,7], k = 1

Output: [7]
```

```text
Input: nums = [4,1,-1,2,-1,2,3], k = 2

Output: [-1,2]
```

**Code**

```python
from collections import Counter, defaultdict

def top_k_elements(nums: List[int], k: int)-> List[int]:
        n = len(nums)
        counts = Counter(nums)
        buckets = defaultdict(list)
        for num, count in counts.items():
            buckets[count].append(num)

        res = []
        for count in range(n, 0, -1):
            if k == 0:
                break
    
            if count in buckets:
                res.extend(buckets[count])
                k -= len(buckets[count])
        return res
```

If we take the example input `nums = [4,1,-1,2,-1,2,3], k = 2` then we have that:

```python
counts = {4: 1, 1: 1, -1: 2, 2: 2, 3:1}
buckets = {1: [1, 3, 4], 2: [-1, 2]}
```

The intuition is that the buckets contain the count as the key, and a list of elements with that count as the value. Creating the counts and buckets hash maps only takes \\(O(n)\\) time. We also will not use more than \\(O(n)\\) space as the largest key of the buckets dict is \\(n\\) if all elements in nums are the same, and all other keys are less than this value. All values of the buckets dict are just elements of nums, so we also cannot be larger than \\(O(n)\\) space due to the values of the buckets dict. Given the buckets dict, we can now get the top \\(k\\) elements by starting with the highest possible count and in decreasing order of counts, adding elements to our resulting list until we have \\(k\\) elements.

**Time & Space Complexity**

- Time complexity: \\(O(n)\\)
- Space complexity: \\(O(n)\\)

### Prefix sums

Given an integer array `nums`, return an array output where `output[i]` is the product of all the elements of `nums` except `nums[i]`. The idea is to build two arrays of prefix / suffix products as the constraint is usually that you cannot take the product of all elements in `nums` as it is not guaranteed to fit into a 32-bit int.

**Examples**

```text
Input: nums = [1,2,4,6]

Output: [48,24,12,8]
```

```text
Input: nums = [-1,0,1,2,3]

Output: [0,-6,0,0,0]
```

**Code**

```python
def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)

    prefix = [1] * n
    for i in range(1, n):
        prefix[i] = nums[i-1] * prefix[i-1] 

    suffix = [1] * n
    for i in range(n-2, -1, -1):
        suffix[i] = nums[i+1] * suffix[i+1]

    res = []
    for i in range(n):
        res.append(prefix[i] * suffix[i])
    return res
```

This has Time & Space Complexity:

- Time complexity: \\(O(n)\\)
- Space complexity: \\(O(n)\\)

We can further reduce the space complexity by building the result in place:

```python
def product_except_self(nums: List[int]) -> List[int]:
    n = len(nums)
    res = [1] * n

    prefix = 1
    for i in range(1, n):
        prefix = nums[i-1] * prefix
        res[i] *= prefix

    suffix = 1
    for i in range(n-2, -1, -1):
        suffix = nums[i+1] * suffix
        res[i] *= suffix

    return res
```

This would then have Time & Space Complexity:

- Time complexity: \\(O(n)\\)
- Space complexity: \\(O(1)\\) (as typically input / output are not counted in space complexity analysis, so we get to ignore the \\(O(n)\\) sized output array we use).
