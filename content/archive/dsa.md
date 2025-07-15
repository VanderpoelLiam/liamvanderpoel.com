---
title: "Data Structures and Algorithms"
date: 2025-04-08T11:05:46+02:00
draft: false
---
{{< katex >}}

This is a cheatsheet I made to help me prepare for LeetCode-style coding interviews. There is no replacement for just doing lots of practices problems. The problems sets I recommend are the [Neetcode 150](https://neetcode.io/roadmap) or [Blind 75](https://www.techinterviewhandbook.org/grind75/).

Personally I do not think it is worth practicing LeetCode hard problems as the goal during the practice phase is to be exposed to as many different techniques as possible in as little time as possible. Hard problems have a poor "time spent on the problem" to "new technique exposure" ratio. However as a means of checking your understanding of a given technique they can be worthwhile, but only after you have completed all the easy / medium problems in [Neetcode 150](https://neetcode.io/roadmap), [Blind 75](https://www.techinterviewhandbook.org/grind75/), or an equivalent problem set.

## Data Structures

### Lists

Take two lists `a = [1, 3, 2]` and `b = [200, 300, 100]`.

#### Sorting

We can use the [sorted()](https://docs.python.org/3/library/functions.html#sorted) function to sort a list and create a new object:

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

Otherwise we can use the [list.sort()](https://docs.python.org/3/library/stdtypes.html#list.sort) function to sort in-place:

```python
a.sort()
>>> print(a)
[1, 2, 3]

a.sort(reverse=True)
>>> print(a)
[3, 2, 1]

c = list(zip(a, b))
c.sort()
>>> print(c)
[(1, 200), (2, 100), (3, 300)]

c.sort(key=lambda x: x[1])
>>> print(c)
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

#### Fast and Slow Pointers (Floyd's Cycle Detection)

This algorithm is used for linked list related problems to find a cycle. You have two pointers `fast` and `slow`, we increment `slow` by one step each iteration and `fast` by two steps each iteration. The intuition is that the fast pointer will always catch the slow pointer if a cycle exists. This follows from imagining the slow/fast pointers as two runners on a track running in circles. Intuitively the fast runner will always catch the slow one (given they are running in a loop). But why will this take \\(O(n)\\) time? Let `d` be the size of the gap between the fast and slow pointers. Each iteration, the new gap becomes `d + 1 - 2 = d - 1` as the slow pointer increases the gap by one, and the fast pointer decreases it by 1. This means they will be equal in `d` iterations, and as the longest cycle has length \\(O(n)\\), they will be equal in \\(O(n)\\) time. The algorithm then follows:

```python
def cycle_exists(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True
    
    return False
```

#### Find duplicate number in array

A special use case of the fast/slow pointer algorithm is to solve the following problem:

Given an array `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive, there is only one repeated number, but it could be repeated more than once. Find that duplicate number using \\(O(1)\\) space and without modifying the array.

Consider instead that we want to find the node at the start of a cycle in a linked list. For example, consider the list `head -> 5 -> 4 -> 1 -> 3 -> 2 -> 1`, we are looking for the node with value `1`. Running [Floyd's algorithm for cycle detection](https://en.wikipedia.org/wiki/Cycle_detection#Tortoise_and_hare) (our code from above), if the fast/slow nodes meet we must be at some point in the cycle but not necessarily at the start. Let \\(x_0, x_1, ..., x_i\\) be all the nodes traversed by the slow pointer. When the fast and slow pointer meet at \\(x_i\\), we are at the minimum \\(i\\) such that \\(x_i = x_{2i}\\). Let \\(\mu\\) be the number of steps from \\(x_0\\) to the start of the cycle and \\(\lambda\\) be the length of the cycle. Therefore \\(i = \mu + a \cdot \lambda\\) and \\(2i = \mu + b \cdot \lambda\\), where \\(a, b\\) are integers representing how many times the slow/fast pointers looped around the cycle before meeting. Combining both equations we have \\(i = (b - a) \cdot \lambda\\), this means that regardless what node I start at in the cycle, if I take \\(i\\) steps I will loop back around to my current position. So in particular \\(x_{i+\mu} = x_\mu\\). This means if I leave the slow pointer at \\(x_i\\), move the fast pointer to \\(x_0\\) and take \\(\mu\\) steps with both pointers, I will have that the slow pointer is at \\(x_{i+\mu}\\) and the fast pointer is at \\(x_{\mu}\\), but as \\(x_{i+\mu} = x_\mu\\) we have `slow == fast` once again. Consequently if I move the fast pointer to \\(x_0\\) and just increment both pointers until they are equal, this occurs at node  \\(x_\mu\\) which is precisely the node we are looking for. In Python this would look like:

```python
# This assumes a cycle exists.

def find_start_cycle(head):
    slow, fast = head, head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    
    fast = head
    while fast != slow:
        fast = fast.next
        slow = slow.next
  
    return slow
```

Coming back to our original problem of finding repeated elements in an array. We want to reformulate this problem as the start of a cycle in a linked list. Let `nums = [1, 3, 4, 2, 2]`. Let each value in the array be a pointer to the index of the next node, that is for node `i` in the linked list we have `Node(i, nums[i])`, e.g. the first node is `Node(0, 1)` drawn as `0 -> 1`, the second node is `Node(1, 3)` drawn `1->2` (as the value of Node `3` is `2`), etc... If we look at the following table we can see more clearly how to construct the linked list:

```text
index: [0, 1, 2, 3, 4]
value: [1, 3, 4, 2, 2]
```

For each index `i` in `nums`, the node at that position has value `i` and next node at index `nums[i]`, so the linked list would look like:

```text
0 -> 1 -> 2 -> 4
          ^___ |
```

Therefore we simply need to adjust `find_start_cycle(head)` to navigate to the next node by replacing `node = node.next` with `node = nums[node]`. Hence our new algorithm to find duplicates becomes:

```python
def find_duplicate(nums):
    slow, fast = 0, 0

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    fast = 0
    while fast != slow:
        fast = nums[fast]
        slow = nums[slow] 
  
    return slow
```

#### Doubly Linked Lists

[OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict) is like a normal dict in Python but maintains the ordering of elements added to the dict. However since Python 3.7 the in built [dict](https://docs.python.org/3/library/stdtypes.html#dict) class remembers insertion order. The main convenience of the OrderedDict object is the [move_to_end()](https://docs.python.org/3/library/collections.html#collections.OrderedDict.move_to_end) method which will move the key and its associated value to the rightmost (last) position:

```python
d = OrderedDict.fromkeys('abcde')
d.move_to_end('b')
''.join(d) # 'acdeb'

d.move_to_end('b', last=False)
''.join(d) #'bacde'
```

The operations and runtime are the same as for a `dict` with a few differences. See [Hash Maps / Sets](#hash-maps--sets) for more details on the base operations.

| Operation                   | Python Code                              | Time Complexity |
|----------------------------|-------------------------------------------|-----------------|
| Create empty OrderedDict   | `from collections import OrderedDict`<br>`od = OrderedDict()` | O(1)            |
| Create OrderedDict with items | `od = OrderedDict([('a', 1), ('b', 2)])` | O(n)            |
| Pop item from end/start    | `od.popitem(last=True)`                   | O(1)            |
| Move key to end/start      | `od.move_to_end('key', last=True)`        | O(1)            |

[OrderedDict is implemented as a doubly-linked list combined with a hash table](https://github.com/python/cpython/blob/1f0a294e8c2ff009c6b74ca5aa71da6269aec0dd/Lib/collections/__init__.py#L89) to achieve constant-time `popitem(last=True)` and `move_to_end('key', last=True)` operations.

### Queues

Queues are FIFO (First In, First Out). Use `deque` implementation from `collections`, do `push(), pop()` using `append(), popleft()`.

```python
from collections import deque

queue = deque()

queue.append(1) # queue = deque([1])
queue.append(2) # queue = deque([1, 2])

queue.popleft()  # 1, queue = deque([2])

queue = deque([1, 2])  # Initialize deque with [1, 2]
```

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

### Binary Trees

A binary tree is a tree where each node has up to two children. A binary search tree (BST) is a binary tree with the additional invariant that the all nodes in the left subtree are less the parent and all nodes in the right subtree are greater than the parent. For example:

```text
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
```

#### Finding Lowest Common Ancestor

The Lowest Common Ancestor (LCA) of two nodes `p` and `q` in a binary tree is the lowest node (i.e. deepest in tree) that is an ancestor of both. Assume both `p` and `q` are in the tree, and the values are all unique for both algorithms below. If we are dealing with a binary tree (i.e. not a BST) the approach is to traverse the tree starting from the root. If the root is equal to `p` or `q` then the root is the LCA. Otherwise we have three cases:

1. `p` and `q` are in different subtrees, so the root is the LCA
2. `p` and `q` are in the left subtree, so the LCA is the result of recursing on the left subtree.
3. `p` and `q` are in the right subtree, so the LCA is the result of recursing on the right subtree.

The trick is that we return None if we cannot find `p` or `q` in our recursion. For example, if `left = lca(root.left, p, q)` is None this means that `p` and `q` are NOT in the left subtree, so the solution is `right = lca(root.right, p, q)`. This leads to the algorithm:

```python
def lca(root, p, q):
    if root is None or root.val == p.val or root.val == q.val:
        return root

    left = lca(root.left, p, q)
    right = lca(root.right, p, q)

    if left and right:
        # p and q in different subtrees
        return root
 
    # return result on whichever subtree contains p and q
    return left if left else right
```

Time complexity is \\(O(n)\\) is we have to explore both subtrees in the worst case, which requires visiting all nodes. Our space complexity is the recursion depth which is \\(O(h)\\).

For a BST, the invariant allows us to know whether the LCA is in the left/right subtree without having to recurse on both subtrees. The three cases are:

1. `p.val` and `q.val` are less than `root.val`, so the LCA is in the left subtree
2. `p.val` and `q.val` are greater than `root.val`, so the LCA is in the right subtree
3. `p` and `q` are in different subtrees, so the root is the LCA

This leads to a similar recursion as before, but the main difference is never have to explore both subtrees, either the left or the right. This leads a faster algorithm as the runtime depends only on the height of the tree. The space complexity is unchanged:

```python
def lca(root, p, q):
    if root is None:
        return root
    
    if p.val < root.val and q.val < root.val:
        return lca(root.left, p, q)
    
    if p.val > root.val and q.val > root.val:
        return lca(root.right, p, q)
    
    return root
```

#### Inorder, Preorder, and Postorder Traversal of a Binary Tree

If we are doing a DFS traversal of a binary tree we have a choice on when we visit the node during our recursion. If we consider the DFS algorithm below we can see the three possible choices we have for when to visit the root:

```python
def dfs(root):
    if not root:
        return
    
    # Can visit root here -- Preorder
    dfs(root.left)
    # Or can visit root here -- Inorder
    dfs(root.right)
    # Lastly can visit root here -- Postorder
```

In short:

| Traversal | Visit Order           |
|----------------|------------------------|
| Inorder        | Left Subtree → Root → Right    |
| Preorder       | Root → Left Subtree → Right Subtree    |
| Postorder      | Left Subtree → Right Subtree → Root    |

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

### Heaps and Priority Queues

A priority queue is an abstract data structure that is an array where we each element has a priority and we always pop the highest priority element first regardless of when it was added to the priority queue. A heap is an efficient implementation of a priority queue. It is a complete binary tree satifying the heap property: for every node, the value of its children is greater than or equal to its own value. An example min heap:

```text
        2
      /   \
     4     5
    / \   /
   10  9  6
```

In python the [heapq library](https://docs.python.org/3/library/heapq.html) implements a min-heap (there is no max-heap implementation), so the above heap would be represented as:

```python
import heapq

nums = [10, 9, 6, 2, 4, 5]

# In-place transformation of nums into a min-heap
heapq.heapify(nums)

# Push a new element into the heap
heapq.heappush(nums, 1)

# Pop the smallest element from the heap
heapq.heappop(nums)

# Peek at min value in heap
nums[0]
```

The time complexity of the heap operations are:

| Operation | Python | Time Complexity |
|-----------------------------|--------------------------------------|-----------------|
| Create heap from list       | `heapq.heapify(nums)`               | O(n), Yes linear not O(n log n) see [docs](https://docs.python.org/3/library/heapq.html#heapq.heapify)         |
| Push item                   | `heapq.heappush(nums, item)`        | O(log n)        |
| Pop smallest item           | `heapq.heappop(nums)`               | O(log n)        |
| Peek at smallest item       | `nums[0]`                            | O(1)            |

#### Priority Queue Example - K Closest Points to Origin

Suppose we are given a list of `points` on the 2D plane and we want to efficiently return the `k`'th closest points to the origin. We would want to initialize a priority queue using a min-heap where the priority is distance to the origin. What entries should we add to our heap to ensure the closest point to the origin is a the top of the heap? The implementation details are explained in [Priority Queue Implementation Notes](https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes), but in short add entries of the form `[priority, value]` to the heap. For example in our case we would store `[dist to origin, x, y]` in our min-heap, implemented as:

```python
import math
import heapq

def distance_to_origin(x, y):
    return math.sqrt(x**2 + y**2)

min_heap = []
for x, y in points:
    dist = (x ** 2) + (y ** 2)
    min_heap.append([dist, x, y])

heapq.heapify(min_heap)
```

#### Kth Largest Element in an Array

There is a trick to using a heap to find the `k`'th largest element in an array `nums` using a heap. There are two approaches depending if we use a min-heap or a max-heap:

##### Min-Heap approach

The idea is that we want the min-heap to contain the `k`'th largest elements seen so far. This is achieved by iteratively adding each number in nums to the min-heap and popping the root each time the list grows larger than `k`. After we have iterated over nums, the min-heap contains the `k` largest elements in nums in increasing order:

```python
def find_kth_largest(nums, k):
    min_heap = []
    heapq.heapify(min_heap)

    for n in nums:
        heapq.heappush(min_heap, n)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0]
```

Let `n = len(nums)`. The time complexity is \\(O(n \cdot \log{k})\\) as our heap never grows larger than `k` and we do our push/pop trick for each of the `n` numbers. The space complexity is the size of the heap which is \\(O(k)\\).

Alteratively we can use a max-heap. Where we just init the min_heap with negative values, pop the first `k-1` elements in our max_heap, and return the largest element.

```python
def find_kth_largest(nums, k):
    max_heap = [-n for n in nums]
    heapq.heapify(max_heap)

    for i in range(k-1):
        heapq.heappop(max_heap)
    
    return -1 * max_heap[0]
```

The time complexity is now \\(O(n + k \cdot \log{n})\\) as our heap has size `n` and we pop `k-1` elements. The space complexity now depends on `n` as our heap has size  \\(O(n)\\).

### Graphs

TODO

## Algorithms

### Dynamic Programming

#### Decode Ways  

Consider the following problem:

We encode a string into a sequence of digits using the mapping `"A" -> "1"`,  `"2" -> "2"` ... `"Z" -> "26"`. Given a string `s` containing only digits, return the number of ways to decode it. E.g. Given `s = "1012"` we can decode it as `10, 1, 2 = "JAB"` or `10, 12 = "JL"`.

To solve this problem we need to understand the decision tree we are trying to explore. We are interested in partitions of `s` that are subject to some constraint. For `s = "1012"` and our decoding constraint the tree looks like:

```text
                    ""
            "1"               "10"
        "0"    "01"        "1"    "12"
                        "2"
```

At each split, we can either take a substring of length 1 or 2 subject to the constraint that the substring is a valid encoding. Formally a substring `substr` is a valid encoding if either condition holds:

- `len(substr) == 1` and `substr != "0"`
- `len(substr) == 2` and `substr[0] != "0"` and `int(substr) <= 26`

##### Recursive Solution

The first approach to a dynamic programming problem is the brute force recursive solution. In this case we want to follow a path in the decision tree until we reach a leaf, increase our count, then backtrack.

```python
def num_decodings(s):
    n = len(s)
    count = 0

    def dfs(i):
        if i > n:
            return 

        if i == n:
            count += 1
        
        if s[i:i+1] != "0":
            dfs(i+1)
        
        if i < n-1 and s[i:i+1] != "0" and int(s[i:i+2]) <= 26:
            dfs(i+2)

    dfs(0)
    return count
```

##### Top-down

Next, we want to avoid repeated work. For example with `s = "1012"` the recursion stack looks like:

```text
dfs(5)
dfs(4)
dfs(5)
dfs(4)
dfs(3)
dfs(2)
dfs(1)
dfs(0)
```

So we are repeating the `dfs(5)` and `dfs(4)` calls. To have a top-down dynamic program we want to memorize the solution at each index to avoid repeated work. To start it is necessary to rewrite the recursive solution to not rely on global variables (this is not always needed but often keeping the global leads to issues adding memoization).

```python
def num_decodings(s):
    n = len(s)

    def dfs(i):
        if i > n:
            return 0

        if i == n:
            return 1
        
        count = 0
        if s[i:i+1] != "0":
            count += dfs(i+1)
        
        if i < n-1 and s[i:i+1] != "0" and int(s[i:i+2]) <= 26:
            count += dfs(i+2)
        return count

    return dfs(0)
```

Then we can easily adapt it by adding a 1D array `dp` to store solutions we have seen before:

```python
def num_decodings(s):
    n = len(s)
    dp = [-1] * n

    def dfs(i):
        if i > n:
            return 0

        if i == n:
            return 1
        
        if dp[i] != -1:
            return dp[i]
        
        dp[i] = 0
        if s[i:i+1] != "0":
            dp[i] += dfs(i+1)
        
        if i < n-1 and s[i:i+1] != "0" and int(s[i:i+2]) <= 26:
            dp[i] += dfs(i+2)
        return dp[i]
    
    return dfs(0)
```

If there are not nice bounds on you solution (e.g. here we know the count is always non-negative), rather than using an array with an out of bounds value to indicate the problem is not yet solved, we can use an dict:

```python
def num_decodings(s):
    n = len(s)
    dp = {} 

    def dfs(i):
        #[... same as before ...] 
        
        if i in dp:
            return dp[i]
        
        #[... same as before ...]
    
    return dfs(0)
```

##### Bottom-up

A bottoms-up approach builds the solution iteratively. The idea is that our recursion is \\(dp[i] = 1_{s[i:i+1] \\text{ is valid}} \\cdot dp[i+1] + 1_{s[i:i+2] \\text{ is valid}} \\cdot dp[i+2]\\), so we iterate from `n-1` to `0` inclusive with the base cases being `dp[n] = 1`.

```python
def num_decodings(s):
    n = len(s)
    dp = [0] * (n+1)
    dp[n] = 1

    for i in range(n-1, -1, -1):
        dp[i] = 0
        if s[i:i+1] != "0":
            dp[i] += dp[i+1]
        
        if i < n-1 and s[i:i+1] != "0" and int(s[i:i+2]) <= 26:
            dp[i] += dp[i+2]

    return dp[0]
```

##### Space Optimized

Lastly we can optimize the space usage of our dp table by noticing that at index `i` we only need to store the solution for `i+1` and `i+2`:

```python
def num_decodings(s):
    n = len(s)
    temp = 0
    dp1 = 1 # i+1
    dp2 = 0 # i+2

    for i in range(n-1, -1, -1):
        if s[i:i+1] != "0":
            temp += dp1
        
        if i < n-1 and s[i:i+1] != "0" and int(s[i:i+2]) <= 26:
            temp += dp2
        
        dp2 = dp1
        dp1 = temp 
        temp = 0

    return dp1
```

##### Time and Space Analysis

The time/space analysis for each approach is given below:

| Approach                   | Time Complexity | Space Complexity | Explanation                                                                 |
|----------------------------|-----------------|------------------|-----------------------------------------------------------------------|
| Recursive Brute Force      | O(2^n)           | O(n)             | Explores entire decision tree, see [Backtracking](#backtracking) for more info   |
| Top-Down DP (Memoization)  | O(n)            | O(n)             | Caches intermediate results during recursion with memoization          |
| Bottom-Up DP (Tabulation)  | O(n)            | O(n)             | Iteratively fills up dp table                    |
| Space-Optimized Bottom-Up  | O(n)            | O(1)             | Optimizes tabulation by effectively having a constant size dp table    |

#### Maximum Product Subarray

Given an integer array nums, find a subarray that has the largest product within the array and return it. We don't allow empty subarrays, and do allow negative integers. Here are two approaches whose patterns are applicable to other problems:

##### Kadane's Algorithm

If we don't have products but rather have sums, at each index i we can:

1. Start a new subarray of only `nums[i]`
2. Extend the current subarray with `nums[i]`

So we need to track the maximum sum ending at index `i` as well as the overall result. This leads to the algorithm:

```python
def max_sum_subarray(nums):
    n = len(nums)
    max_sum = res = nums[0]

    for i in range(1, n):
        max_sum = max(nums[i], max_sum + nums[i])
        res = max(res, max_sum)
    return res
```

For products the idea is the same, except that we also need to track the minimum product so far as a negative `nums[i]` can flip the min/max products. Formally this is the max/min product subarray ending at index `i`.

```python
def max_product_subarray(nums):
    n = len(nums)
    res = max_prod = min_prod = nums[0]

    for i in range(1, n):
        if nums[i] < 0:
            # Swap max/min products
            max_prod, min_prod = min_prod, max_prod

        max_prod = max(nums[i], nums[i] * max_prod) 
        min_prod = min(nums[i], nums[i] * min_prod) 
        res = max(res, max_prod)

    return res
```

##### Prefix and Suffix sums

An alternate approach is to track the running product iterating over the array in both directions. The edge case to be careful about is that if the prefix/suffix is ever zero, we need to reset the running product to the current element. Consider `nums = [1,2,-3,4]` as to why we need to look at both directions, where `prefix[i] = prod(num[:i+1])` and `suffix[i] = prod(nums[(n-1)-i:])`:

```python
prefix = [1, 2, -6, -24]
suffix = [4, -12, -24, -24]
```

So we would miss the max product if we only scanned in one direction.

```python
def max_product_subarray(nums):
    n = len(nums)
    max_prod = nums[0]
    prefix = suffix = 1

    for i in range(n):
        if prefix == 0:
            prefix = 1
       
        if suffix == 0:
            suffix = 1

        prefix *= nums[i]
        suffix *= nums[(n-1) - i]

        max_prod = max(max_prod, prefix, suffix)

    return max_prod
```

#### Longest Increasing Subsequence

Consider the problem: Given an integer array nums, return the length of the longest strictly increasing subsequence. Recall a subsequence is a subset created by deleting zero or more elements e.g. `[1, 2, 3]` is a subsequence of `nums = [9,1,4,2,3,3,7]`.

The difficulty in this problem is that the natural recursive solution does not naturally lead to a dynamic programming solution. Take `nums = [1, 0, 2]` where the LIS is `[1,2]` or `[0,2]` and they both have length `2`. The decision tree is to consider all subsequences that are strictly increasing, at each level `i` we either include or don't include `nums[i]`:

```text
                        []
            [1]                   []               i = 0
               [1]          [0]        []          i = 1
           [1,2]  [1]   [0,2]  [0]  [2]   []       i = 2
```

The natural DFS solution to explore this decision tree is:

```python
def length_of_LIS(nums):
    n = len(nums)

    def dfs(i, subseq):
        if i == n:
            return len(subseq)
        
        res = dfs(i+1, subseq)

        if not subseq or subseq[-1] < nums[i]:
            res = max(res, dfs(i+1, subseq + [nums[i]]))

        return res
    
    return dfs(0, []) 
```

As there are \\(2^n\\) paths, and we compute the length of the subsequence at the end of each path this has runtime \\(O(n\cdot 2^n)\\). We can be slightly more efficient with runtime (O(2^n)\\) if we update the length of the subsequence as we go, but this does not nicely lead to a DP solution:

```python
def length_of_LIS(nums):
    n = len(nums)

    def dfs(i, len_subseq, max_elem):
        if i == n:
            return len_subseq
        
        res = dfs(i+1, len_subseq, max_elem)

        if max_elem < nums[i]:
            res = max(res, dfs(i+1, 1 + len_subseq, nums[i]))

        return res
    
    return dfs(0, 0, -math.inf) 
```

Instead we need to rewrite our decision tree to represent the state in a way that is easier to memoise, and track the length of the path as we explore the decision tree rather than at the end. We have that `dfs(i,j)` is the length of the longest strictly increasing subsequence starting at index `i` and with index `j` the last element of the subsequence (therefore also the maximum value seen so far). Below is the same decision tree as above for  `nums = [1, 0, 2]`, but we provide both the subsence and its `i,j` representation:

```text
                                 []=(0, -1)
                [1]=(1,0)                                 []=(1,-1)               
                    [1]=(2,0)                  [0]=(2,1)              []=(2,-1)          
            [1,2]=(3,2)  [1]=(3,0)   [0,2]=(3,2)  [0]=(3,1)  [2]=(3,2)   []=(3,-1)       
```

This leads to the recursive solution:

```python
def length_of_LIS(nums):
    n = len(nums)

    def dfs(i, j):
        if i == n:
            return 0
        
        res = dfs(i+1, j)

        if j == -1 or nums[j] < nums[i]:
            res = max(res, 1 + dfs(i+1, i))

        return res
    
    return dfs(0, -1) 
```

With this representation we more clearly see the repeated work e.g. (3,2) appears 3 times. To avoid this repeated work with memoisation we notice that there are `n+1` choices for `i` and `n` choices for `j`. This leads to the algorithm:

```python
def length_of_LIS(nums):
    n = len(nums)
    dp = [[-1]*n]*(n+1)

    def dfs(i, j):
        if i == n:
            return 0
        
        if dp[i][j] != -1:
            return dp[i][j]

        res = dfs(i+1, j)

        if j == -1 or nums[j] < nums[i]:
            res = max(res, 1 + dfs(i+1, i))

        dp[i][j] = res
        return res
    
    return dfs(0, -1) 
```

This has both time and space complexity of \\(O(n^2)\\). We can optimize the solution further by iterating through the array backwards and storing the solution starting at index `i` in `dp[i]`:

```python
def length_of_LIS(nums):
    n = len(nums)
    dp = [1] * n
        
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if nums[j] > nums[i]:
                dp[i] = max(dp[i], 1 + dp[j])
    
    return max(dp)
```

This leads to a pretty optimal solution with time complexity \\(O(n^2)\\) and space complexity \\(O(n)\\).

### 2D Dynamic Programs

A important mistake not to make when initializing a 2D dynamic program in Python is say we want a `m x n` grid. This is correct:

```python
dp = [[-1] * n for _ in range(m)]
```

while this leads to all rows pointing to the same object in memory:

```python
dp = [[-1] * n] * m
```

We can test this with the following code:

```python
>>> m = 3
>>> n = 4
>>> dp = [[-1] * n for _ in range(m)]
>>> dp[0][0] = 99
>>> dp
[[99, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
>>> dp[0] is dp[1]
False

>>> dp = [[-1] * n] * m
>>> dp[0][0] = 99
>>> dp
[[99, -1, -1, -1], [99, -1, -1, -1], [99, -1, -1, -1]]
>>> dp[0] is dp[1]
True
```

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

- `h`: Depth of recursion / number of decisions
- `b`: Branching factor (choices per step)

**Time Complexity  \\(O(b^h)\\):**

In the worst case, we explore all possible combinations of `b` choices at `h` levels:

```text
Level 0:        1 call
Level 1:        b calls
Level 2:        b^2 calls
…
Level h:        b^h calls
```

So in total we have \\(1 + b + b^2 + \dots + b^h\\). This is the sum of a geometric series which equals  \\(\frac{b^{h+1} - 1}{b - 1} = O(b^h)\\)

**Space Complexity \\(O(h)\\):**

The space required during execution is made up of the call stack and the current state. The call stack has size \\(O(h)\\) as it is can be up to the size of the maximum recursion depth. The size of the state depends on the specifics of the problem, but it is also usually of size \\(O(h)\\). This gives the total space complexity of \\(O(h)\\).

#### Finding all subsets

Let's look at a specific example of backtracking. Given an array nums of unique integers, return all possible subsets of nums. For example if `nums = [1,2,3]` we would return `[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]`. For each element of nums we include it or don't include it in a subset. This leads to the following decision tree, and our algorithm is to perform a DFS traversal of the tree and add the value of each node to our result.

```text
                              []                # Decision: Include nums[0]
                      /               \
                    [1]               []        # Decision: Include nums[1] 
                 /      \           /     \
            [1,2]       [1]        [2]    []    # Decision: Include nums[2]
            /   \       / \        / \    / \
       [1,2,3] [1,2] [1,3] [1] [2,3] [2] [3] []
```

We need to track the subset we have constructed so far, as well as the index we are making a decision on. We are at a leaf node of our decision tree if the index we are considering is greater than `len(nums)`, at this point we can append our current subset to the solution. This leads to the algorithm:

```python
def subsets(nums):
    res = []
    n = len(nums)

    def backtrack(subset, i):
        if i > n:
            # Copy necessary in Python as subset is passed by reference
            # For more details see: https://nedbatchelder.com/text/names.html
            res.append(subset.copy())
        
        
        # Include element nums[i]
        subset.append(nums[i])
        backtrack(subset, i+1)

        # Don't include element nums[i]
        subset.pop()
        backtrack(subset, i+1)

    backtrack([], 0)
    return res
```

#### Palindrome Partitioning

Suppose you want to partition a string `s` into substrings where each substring is a palindrome. For examples, for `s = "aab"` we would output `[["a","a","b"],["aa","b"]]`. The approach is that at each start index of `s` we can take the first 1 char, or the first 2 chars, ... or the first `n-i` to be our partition. This is visualized below:

```text
                       []
               /        |       \
           [a]         [aa]      [aab]
         /    \         |
     [a,a]   [a,ab]   [aa,b]
      /      
 [a,a,b]   
```

Then we only need to add the constraint to exclude all solutions where we consider a substring that is not a palindrome:

```python
def is_palindrome(s):
    return s == s[::-1]

def partition(s):
    n = len(s)
    res = []
    soln = []

    def backtrack(start):
        if start >= n:
            res.append(soln.copy())
            return

        for end in range(start + 1, n + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                soln.append(substring)
                backtrack(end)
                soln.pop()

    backtrack(0)
    return res
```

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

### Graph traversal

All BFS and DFS implementations hav Time & Space Complexity:

- Time complexity: \\(O(V + E)\\), we have to check all nodes and edges.
- Space complexity: \\(O(V)\\), we have to track visited nodes.

#### Breadth-First Search (BFS)

This traversal explores explores all neighbors level by level in a graph. It uses a queue to track the next nodes to visit and is commonly used to find the closest vertex to the start satisfying some conditions e.g. shortest path between two vertices. It implemented iteratively as follows:

```python
from collections import deque

def bfs(root):
    visited = set(root)
    queue = deque([root])
    
    while queue:
        v = queue.popleft()
        print(v)

        for u in neighbours(v):
            if u not in visited:
                visited.add(u)
                queue.append(u)
```

#### Depth-First Search (DFS)

DFS visits the child vertices before visiting the sibling vertices. It uses a stack to track the next nodes to visit and as such the iterative implementation is the exact same as for BFS but with a stack instead of a queue. As `deque` can also be used as a stack, we only have to change a single line `v = queue.popleft()` to `v = queue.pop()`, however it is common to use a list instead:

```python
def dfs_iterative(root):
    visited = set(root)
    stack = [root]
    
    while stack:
        v = stack.pop()
        print(v)

        for u in neighbours(v):
            if u not in visited:
                visited.add(u)
                stack.append(u)
```

The recursive implementation of DFS

```python
def dfs_recursive(root):
    def dfs(root, visited):
        if root not in visited:
            print(root)
            visited.add(root)
            for u in neighbours(v):
                dfs(u, visited)

    return dfs(root, set())
```