---
title: "Data Structures and Algorithms"
date: 2025-04-08T11:05:46+02:00
draft: false
tableOfContents:
  endLevel: 2
---
{{< katex >}}

## Data Structures

TODO.

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

The intuition is that the buckets contain the count as the key, and a list of elements with that count as the value. Creating the counts and buckets hash maps only takes \\(O(n)\\) time. We also will not use more than \\(O(n)\\) space as the largest key of the buckets dict is \\(n\\) if all elements in nums are the same, and all other keys are less than this value. All values of the buckets dict are just elements of nums, so we also cannot be larger than \\(O(n)\\) space due to the values of the buckets dict. Given the buckets dict, we can now get the top \\(k\\) elements by starting with the highest possible count and in decreasing order of counts, adding elements to our resultiing list until we have \\(k\\) elements.

**Time & Space Complexity**

- Time complexity: \\(O(n)\\)
- Space complexity: \\(O(n)\\)
