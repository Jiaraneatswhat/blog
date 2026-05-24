---
title: "😎 LeetCode 1. 两数之和 [哈希表]"
published: 2026-05-24 08:00:00
category: LeetCode
image: "./cover.png"
---

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出和为目标值 `target` 的那两个整数，并返回它们的数组下标。假设每种输入只会对应一个答案，不能使用两次相同的元素。

:::note[示例 1:]
**输入**: `nums = [2,7,11,15]`, `target = 9`<br>
**输出**: `[0,1]`
:::

:::note[示例 2:]
**输入**: `nums = [3,2,4]`, `target = 6`<br>
**输出**: `[1,2]`
:::

:::note[示例 3:]
**输入**: `nums = [3,3]`, `target = 6`<br>
**输出**: `[0,1]`
:::

### 思路

- 创建一个哈希表，将数组元素值作为 `key`，索引作为 `value`；
- 遍历数组，对于每个元素 `num_i`，检查 `target - num_i` 是否存储在哈希表中，是则输出当前元素索引和 `target - num_i` 的索引，否则将 `num_i` 存入哈希表

### 题解
#### 1. python
```python
class Solution: 
	# 加装饰器实现静态方法
    @staticmethod
    def twoSum(nums, target):  
        hashtab = dict()  
        for ind, num in enumerate(nums):  
            if target - num in hashtab:  
                return [ind, hashtab[target - num]]  
            hashtab[num] = ind  
        return []
```

#### 2. java
```java
public class Solution {
	static int[] addTwo(int[] nums, int target) {  
	        HashMap<Integer, Integer> hashTab = new HashMap<>();  
	        for (int i = 0; i < nums.length; i++) {  
	            if (hashTab.containsKey(target - nums[i])) return new int[{i, hashTab.get(target - nums[i])};  
	            hashTab.put(nums[i], i);  
	        }  
	        return new int[0];  
	    }
}
```

