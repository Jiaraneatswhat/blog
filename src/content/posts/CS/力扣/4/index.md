---
title: "😡 LeetCode 4. 寻找两个正序数组的中位数 [二分查找]"
published: 2026-06-03 09:00:00
category: CS
image: "./cover.png"
---

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的**中位数**。

算法的时间复杂度应该为 `O(log (m+n))`。

:::note[示例 1:]
**输入**: `nums1 = [1,3], nums2 = [2]`<br>
**输出**: `2.00000`，即 `[1,2,3]` 的中位数
:::

:::note[示例 2:]
**输入**: `nums1 = [1,2], nums2 = [3,4]`<br>
**输出**: `2.50000`
:::

### 思路

- 直接遍历的话复杂度是 $O(m+n)$，$\ce {log}$ 表示需要考虑二分法 
- 有序数组的中位数是该数组中第 $k$ 小的数 (奇数)，或是 $k$ 和 $k + 1$ 的均值，可将问题转化为寻找第 $k$ 小的数
- 每次比较两个数组第 $k/2$ 个数的大小，小的那一方的前半部分肯定更小，可以直接舍弃

### 题解
```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
		totalLen = m + n
		left = (total_len + 1) // 2
		right = (total_len + 2) // 2
		
	def get_kth(a, b, k):

		# 通过最短的数组来进行边界判断
		if len(a) > len(b):
			return get_kth(b, a, k)

		# 直接返回 b 中第 k 小的元素
		if len(a) == 0:
			return b[k - 1]

		i = min(len(a), k // 2)
		j = min(len(b), k // 2)

		# 排除 a 的前半部分
		if(a[i - 1] < b[j - 1]):
			# 排除了 i 个数，现在找第 k - i 小的
			return get_kth(a[i:], b, k - i)
		# 大小反过来，或者是相等，都可以调用
		# 相等时排除 a 或 b 等价
		else:
			return get_kth(a, b[j:], k - j)
		
```