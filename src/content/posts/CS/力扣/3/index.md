---
title: "😐LeetCode 3. 无重复字符的最长子串 [指针]"
published: 2026-06-02 08:05:00
category: CS
image: "./cover.png"
---

给定一个字符串 `s` ，请你找出其中不含有重复字符的**最长子串**的长度。

:::note[示例 1:]
**输入**: `s = "abcabcbb"`<br>
**输出**: `3`，即 `length(abc)`
:::

:::note[示例 2:]
**输入**: `s = "bbbbb"`<br>
**输出**: `1``
:::

:::note[示例 2:]
**输入**: `s = "pwwkew"`<br>
**输出**: `3`，即 `length(wke)`
:::

### 思路
- 定义一个空数组用于保存索引
- 通过右指针遍历字符串，将字符取出来
- 若字符不在数组里，则不重复，此时左指针不动，计算长度，然后更新索引
- 若字符在数组里，则更新左指针到 `right + 1`

### 题解 

```python
class Solution:  
    @staticmethod  
    def maxLenSubstr(s):  
        index = [0] * 128  
        maxLen, left = 0  
        for right in range(len(s)):  
            ch = s[right]
			# 遇到重复字符时 left 才会变
            left = max(left, index[ord(ch)])  
            maxLen = max(maxLen, right - left + 1)
            index[ord(ch)] = right + 1  
          
        return maxLen
```
