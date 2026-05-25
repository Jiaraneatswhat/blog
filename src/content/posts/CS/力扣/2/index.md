---
title: "😐 LeetCode 2. 两数相加 [链表]"
published: 2026-05-24 09:00:00
category: CS
image: "./cover.png"
---

给你两个**非空**的链表，表示两个非负的整数。它们每位数字都是按照**逆序**的方式存储的，并且每个节点只能存储**一位**数字。

请你将两个数相加，并以相同形式返回一个表示和的**链表**。

你可以假设除了数字 `0` 之外，这两个数都不会以 `0` 开头。

:::note[示例 1:]
**输入**: `l1 = [2,4,3]`, `l2 = [5,6,4]`<br>
**输出**: `[7,0,8]`，即 `342 + 465 = 807`
:::

:::note[示例 2:]
**输入**: `l1 = [2,4,3]`, `l2 = [2,4,3]`<br>
**输出**: `[0]`
:::

:::note[示例 3:]
**输入**: `l1 = [9,9,9,9,9,9,9]`, `l2 = [9,9,9,9]`<br>
**输出**: `[8,9,9,9,0,0,0,1]`
:::

### 思路

- 链表是逆序存的，`head` 处对应的就是个位数
- 需要返回的是一个新链表，首先对个位数求和，将其结果作为新链表的 `head`
- 在求和时用一个变量 `i` 存储进位，进位非 `0` 即 `1`
- 当 `l1` 和 `l2` 长度相同时，同时进行遍历，求和结果加上进位后 `mod 10`
- 当 `l1` 和 `l2` 长度不相同时，有一方遍历结束后，单独遍历另一方
- 最后检查进位是否计算完毕，否则的话创建一个新的 `val = 1` 的节点接在循环结束的 `node` 后，返回个位数求和得到的 `head` 即可

### 题解

#### 1. python

```python
class Listnode:  
    def __init__(self, val=0, next=None):  
        self.val = val  
        self.next = next  
  
class Solution:  
    @staticmethod  
    def addTwoNum(l1: Listnode, l2: Listnode):  
        # 当前指针在 head 处，处理的是个位的和  
        node1 = Listnode((l1.val + l2.val) % 10)  
        i = (l1.val + l2.val) % 10  
        # 处理完个位后得到一个节点，用于指向后续计算结果  
        node = node1  
        if l1.next != None and l2.next != None:  
            while l1.next and l2.next:  
                # 十位的和要加上上一次的进位 i  
                node.next = Listnode((l1.next.val + l2.next.val + i) % 10)  
                i = (l1.next.val + l2.next.val + i) % 10  
                l1 = l1.next  
                l2 = l2.next  
                node = node.next  
        if l1.next != None:  
            while l1.next:  
                node.next = Listnode((l1.next.val + i) % 10)  
                i = (l1.next.val + i) % 10  
                l1 = l1.next  
                node = node.next  
        if l2.next != None:  
            while l2.next:  
                node.next = Listnode((l2.next.val + i) % 10)  
                i = (l2.next.val + i) % 10  
                l2 = l2.next  
                node = node.next  
        # 如果多出来一个进位， i 仍为 1  
        if i == 1:  
            node.next = Listnode(1)  
        return node1  
      
    @staticmethod  
    def print_linked_list(head):  
        while head:  
            print(head.val, end=" → ")  
            head = head.next  
        print("None")
```

#### 2. java

```java
public class Solution {  
    static ListNode addTwoNum(ListNode l1, ListNode l2) {  
        ListNode head = null, tail = null;  
        int i = 0;  
        while (l1 != null || l2 != null) {  
            int n1 = l1 != null ? l1.val : 0;  
            int n2 = l2 != null ? l2.val : 0;  
            int sum = n1 + n2 + i;  
            // 第一次 head 为 null  
            if (head == null) {  
                head = tail = new ListNode(sum % 10);  
            } else {  
                tail.next = new ListNode(sum % 10);  
                tail = tail.next;  
            }  
            i = sum % 10;  
            if (l1.next != null) l1 = l1.next;  
            if (l2.next != null) l2 = l2.next;  
        }  
        if (i > 0) tail.next = new ListNode(i);  
        return head;  
    }  
}
```


