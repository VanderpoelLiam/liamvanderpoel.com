---
title: "Data Structures and Algorithms"
date: 2025-04-08T11:05:46+02:00
draft: false
---
{{< katex >}}

## Data Structures

### Lists

Take two lists `a = [1, 3, 2]` and `b = [200, 300, 100]`.

#### Sorting

```python
>>> sorted(a)
[1, 2, 3]
>>> sorted(a, reverse=True)
[3, 2, 1]
>>> sorted(zip(a, b))
[(1, 200), (2, 100), (3, 300)]
>>> sorted(zip(a, b), key=lambda x: x[1])
[(2, 100), (1, 200), (3, 300)]
```

#### Slicing

The slicing operation for lists is identical to for strings, see [String slicing](#string-slicing). An important note is that slicing creates a shallow copy of the list. See [the docs](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) for more information.

Common slicing operations for list are:

```python
>>> a[::-1] # Reversing a list
[2, 3, 1]
>>> c = a[:] # Shallow copy list
>>> c[0] = -1
>>> a
[1, 3, 2]
>>> c
[-1, 3, 2]
```

### range()

The function has the form `range(start, stop, step)`, see [docs](https://docs.python.org/2/library/functions.html#range). We go from start with increment step until the next step would exceed (or be less than for negative step) stop. Examples:

```python
>>> list(range(0, 10, 2))
[0, 2, 4, 6, 8]
>>> list(range(0, 10, 6))
[0, 6]
>>> list(range(5, 0, -1))
[5, 4, 3, 2, 1]
```

### String slicing

Similar to range for a string `s` we have `s[start:stop]` that goes includes the start element but not the stop element. Examples:

```python
>>> s = "012345"
>>> s[0:2]
"01"
>>> s[1:4]
"123"
>>> s[3:]
"345"
>>> s[:5]
"01234"
```

Can optionally include a `step` i.e. `s[start:stop:step]` which increments the index by step each time. Examples:

```python
>>> s = "012345"
>>> s[1:4:2]
"13"
>>> s[::2]
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

Python does not have native linked lists, so usually they are defined like:

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

#### Reversing a Linked List

Define linked list class as above, and you now want to reverse the list:

```shell
head -> 0 -> 1 -> 2 -> 3
```

to the list:

```shell
head -> 3 -> 2 -> 1 -> 0
```

The we can do this iteratively:

```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        temp = curr.next
        curr.next = prev
        prev = curr
        curr = temp
    return prev
```

or recursively:

```python
def reverse_list(head):
    if not head:
        return None

    if not head.next:
        return head

    new_head = reverse_list(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

This is a common sub-problem for linked list related problems.

### Stacks

Stacks are LIFO (Last In, First Out). Implemented as lists, do `push(), pop()` using `append(), pop()`.

```python
stack = []

stack.append(1) # stack = [1]
stack.append(2) # stack = [1, 2]

stack.pop()  # 2, stack = [1]

top = stack[-1] # 1, stack = [1]
```

Note that as `deque` is a double-ended queue, we can add/remove from both ends in \\(O(1)\\) time. So we can also implement a stack as:

```python
from collections import deque

stack = deque()

stack.append(1) # stack = deque([1])
stack.append(2) # stack = deque([1, 2])

top_item = stack.pop() # 2, stack = deque([1])

top = stack[-1] # 1, stack = deque([1])
```

### Queues

Queues are FIFO (First In, First Out). Use `deque` implementation from `collections`, do `push(), pop()` using `append(), popleft()`.

```python
from collections import deque

queue = deque()

queue.append(1) # queue = deque([1])
queue.append(2) # queue = deque([1, 2])

queue.popleft()  # 1, queue = deque([2])
```

### Binary Trees

TODO

### Hash Maps / Sets

Sets in Python are implemented as a hash map, so the time complexities are the same for both. The [docs](https://wiki.python.org/moin/TimeComplexity) state that "The Average Case assumes the keys used in parameters are selected uniformly at random from the set of all keys.", so unless we are adversely selecting keys we can assume the average runtime. Note hashmaps and hashtables are the same data structure, called a `dict` in Python.

#### Hash Maps

| Operation                | Python Code              | Time Complexity |
|-------------------------|--------------------------|-----------------|
| Create empty dict       | `d = {}` or `dict()`     | O(1)            |
| Create dict with items  | `d = {'a': 1, 'b': 2}`    | O(n)            |
| Add / Update item       | `d['key'] = value`        | O(1) avg        |
| Remove item by key      | `del d['key']`            | O(1) avg        |
| Check if key exists     | `'key' in d`              | O(1) avg        |
| Access value by key     | `d['key']`                | O(1) avg        |

Examples:

```python
d = {}

d['a'] = 1      # d = {'a': 1}
d['b'] = 2      # d = {'a': 1, 'b': 2}

d['a'] = 42     # Update value, d = {'a': 42, 'b': 2}

d['a']          # 42

'b' in d        # True

d.pop('b')      # Removes and returns 2
```

#### Sets

| Operation                | Python Code            | Time Complexity |
|-------------------------|------------------------|-----------------|
| Create empty set        | `s = set()`            | O(1)            |
| Create set with items   | `s = {1, 2, 3}`        | O(n)            |
| Add item                | `s.add(x)`             | O(1) avg        |
| Remove item             | `s.remove(x)`          | O(1) avg        |
| Check if element exists | `x in s`               | O(1) avg        |

Examples:

```python
s = set()

s.add(1)     # s = {1}
s.add(2)     # s = {1, 2}

s.remove(1)  # s = {2}

2 in s       # True
```

### Heaps

TODO

### Graphs

TODO

## Algorithms

### Sliding Window

The general idea is that we are given an array or a string and we want to find some sub-range that meets some criteria e.g. max sum, longest substring with specific chars, number of distinct elements, etc...

Instead of calculating the criteria for each sub-range which would take \\(O(n^2)\\) time (with \\(n\\) the length of our array/string), we slide a window across our input and update our answer incrementally. The window has left/right pointers and gets expanded or shrunk depending on whether the sub-range under consideration violates some constraint. A running variable is often used to track max, longest etc...

#### Longest Substring Without Repeating Characters

The problem is: Given a string `s`, find the length of the longest substring without duplicate characters.

**Examples:**

```text
Input: s = "zxyzxyz"

Output: 3
```

```text
Input: s = "xxxx"

Output: 1
```

This is a good example of dynamic sliding window. Let `s = "zxyzxyz"`, the idea is to start with a window of the first element `[z]xyzxyz`. The criteria our window needs to satisfy to be valid is that it cannot contain any characters we have seen previously. If it is valid, we update the result tracking the longest substring seen so far and shift our window to the right. Each time we enter the loop, we satisfy the invariant that our previous window was valid, we only need to check is the current character `s[r]` would cause the window to violate the criteria. This would occur at the stage `[zxyz]xyz`, the previous window `[zxy]zxyz` is valid but the current character `s[r] = z` causes the criteria to be invalid. Then we just shift the lhs of the window over until we are valid again. There are additional optimizations that can be done, usually involving using hash maps or sets to speed up checking if an element is in a sub-range, but the key is understanding what the validity criteria of a sub-range is, and how to update our window until it is valid again.

**Code:**

Initial solution:

```python
def longest_substring(s):
    n = len(s)
    l, r = 0, 0
    res = 0

    while r < n:
        # Considering window s[l:r+1], know s[l:r] is 
        # a valid solution
        while s[r] in s[l:r]:
            # Make solution valid again
            l += 1
        
        # Now s[l:r+1] is a valid solution
        res = max(res, len(s[l:r+1]))
        r += 1

    return res
```

Optimized solution:

```python
def longest_substring(s):
    n = len(s)
    l, r = 0, 0
    res = 0
    seen = set()

    while r < n:
        # Considering window s[l:r+1], know s[l:r] is 
        # a valid solution
        while s[r] in seen:
            # Make solution valid again
            seen.remove(s[l])
            l += 1
        
        # Now s[l:r+1] is a valid solution
        res = max(res, r + 1 - l)
        seen.add(s[r])
        r += 1

    return res
```

### Binary Search

Given a array of integers `nums` sorted in increasing order (e.g. `[-1, 0, 2, 3]`) and a target integer `target` we do binary search to find the index of the target as follows:

```python
def binary_search(nums, target):
    n = len(nums)
    l, r = 0, n-1
    while l <= r:
        m = (r+l) // 2
        if nums[m] > target: # 
            r = m - 1
        elif nums[m] < target:
            l = m + 1
        else:
            return m
    return -1
```

Time complexity is \\(O(\log n)\\) as we halve the size of the array under consideration on each iteration of the while loop.

#### Rotated Arrays

Problems where we want to find some value in an increasing array that has been sorted involves a binary search variant where the trick is understanding how to determine if the midpoint is to the right or left of the pivot, or if it is the pivot. As the array is in increasing order, the pivot is also the smallest value. For example, given the array `[1, 2, 3, 4, 5]` the possible rotations are: `[1, 2, 3, 4, 5] -> [5, 1, 2, 3, 4] -> [4, 5, 1, 2, 3] -> [3, 4, 5, 1, 2] -> [2, 3, 4, 5, 1]`.

Take the example `nums = [3, 4, 5, 1, 2]`, and start doing vanilla binary search:

```shell
[3, 4, 5, 1, 2]
 l     m     r
```

The cases are the following:

1. The array has not been rotated, `l` is the pivot.

2. `m` is the pivot, return `m`.

3. `m` is to the left of the pivot, look to the right.

4. `m` is to the right of the pivot, look to the left.

These checks in Python would be the following:

```python
if nums[l] <= nums[r]:
    # array has not been rotated
    return l

# array has been rotated

if nums[m-1] >= nums[m]:
    # m is the pivot
    return m

if nums[l] <= nums[m]:
    # nums[l:m] is an increasing array, so m is to the left of the pivot
    # Look right
    l = m + 1
else:
    # nums[l:m] is not an increasing array, so m is to the right of the pivot
    # Look left
    r = m - 1
```

It is important that the condition is `nums[l] <= nums[m]` and not `nums[l] < nums[m]`. We know `m` is not the pivot, if the sub-array `nums[l:m]` (i.e. including index `l` but not index `m`) is increasing then the pivot is to the right of `m` but this is also the case when `nums[l:m]` is an empty array i.e. when `l=m`. For example when `nums=[2,1]`, we have the initial values:

```shell
[ 2,  1 ]
 l=m  r
```

The sub-array `nums[l:m] = []` is empty, so the pivot cannot be there, hence the pivot is to the right of `m`. 

This is the key trick for problems such as `Find Minimum in Rotated Sorted Array`. For completeness the full algorithm would be for finding the pivot / min value in a rotated increasing array is:

```python
def find_min(nums):
    n = len(nums)
    l, r = 0, n-1
    while l <= r:
        m = (r + l) // 2
        if nums[l] <= nums[r]:
            # nums[l:r] is sorted in increasing order
            return nums[l]

        if nums[m-1] >= nums[m]:
            # We are at the pivot
            return nums[m]

        if nums[l] <= nums[m]:
            # We are to the left of the pivot, go right
            l = m + 1
        else:
            # We are to the right of the pivot, go left
            r = m - 1
```

### Backtracking

General backtracking algorithm is the following:

```python
def backtrack(state):
    if is_goal(state):
        record_solution(state)
        return

    for choice in choices(state):
        if is_valid(choice, state):
            make_choice(choice, state)
            backtrack(state)
            undo_choice(choice, state)
```

**Time & Space Complexity:**

- `n`: Depth of recursion / number of decisions
- `b`: Branching factor (choices per step)

**Time Complexity  \\(O(b^n)\\):**

In the worst case, we explore all possible combinations of `b` choices at `n` levels:

```text
Level 0:        1 call
Level 1:        b calls
Level 2:        b^2 calls
…
Level n:        b^n calls
```

So in total we have \\(1 + b + b^2 + \dots + b^n\\). This is the sum of a geometric series which equals  \\(\frac{b^{n+1} - 1}{b - 1} = O(b^n)\\)

**Space Complexity \\(O(n)\\):**

The space required during execution is made up of the call stack and the current state. The call stack has size \\(O(n)\\) as it is can be up to the size of the maximum recursion depth. The size of the state depends on the specifics of the problem, but it is also usually of size \\(O(n)\\). This gives the total space complexity of \\(O(n)\\).

### Bucket Sort

This algorithm is useful for problems where we want to get the \\(k\\) most frequent elements in a list.

**Examples:**

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

**Code:**

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

**Time & Space Complexity:**

- Time complexity: \\(O(n)\\)
- Space complexity: \\(O(n)\\)

### Prefix sums

Given an integer array `nums`, return an array output where `output[i]` is the product of all the elements of `nums` except `nums[i]`. The idea is to build two arrays of prefix / suffix products as the constraint is usually that you cannot take the product of all elements in `nums` as it is not guaranteed to fit into a 32-bit int.

**Examples:**

```text
Input: nums = [1,2,4,6]

Output: [48,24,12,8]
```

```text
Input: nums = [-1,0,1,2,3]

Output: [0,-6,0,0,0]
```

**Code:**

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
