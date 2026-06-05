---
title: "Transformer-XL 的源码 [LLM]"
published: 2026-06-05 08:00:00
category: CS
image: "./cover.png"
---

这节我们来学习 `Transformer-XL` 的源码，模型的论文内容在之前已有介绍 👉 `taffybook.cn/posts/paper/transformer-xl/` ，其核心类如下:

> 1. 自适应词嵌入 -- `AdaptiveEmbedding`
> 2. 位置编码 -- `PositionalEmbedding`
> 3. 注意力层 -- `RelPartialLearnableMultiHeadAttn`<br>
> &emsp;├── `qkv_net: Linear`           # W_q, W_k,E, W_v（联合投影）<br>
> &emsp;├── `r_net: Linear`             # W_k,R（位置编码投影）<br>
> &emsp;├── `o_net: Linear`             # 输出投影<br>
> &emsp;└── `_rel_shift()`              # 相对位置偏移（关键操作）
> 4. `decoder` -- `RelPartialLearnableDecoderLayer`
> 5. 前馈网络 -- `PositionwiseFF`
> 6. 记忆机制<br>
> &emsp;├── `_init_mems()`                  # 初始化记忆<br>
> &emsp;└── `_update_mems()`                # 更新记忆
> 7. 输出 -- `ProjectedAdaptiveLogSoftmax`

## 1 词嵌入
自适应嵌入可以根据词频动态分配嵌入维度来解决传统嵌入的参数效率问题
- 解决低频词维度过多，高频词维度可能不够的问题

```python
class AdaptiveEmbedding(nn.Module):
	def __init__(
		self, 
		n_token, # 词表大小
		d_embed, 
		d_proj, # 投影后维度
		cutoffs, # 词频分界点
		div_val=1, # 用于维度计算
		sample_softmax=False):
		super().__init__()

		# 成员初始化
		self.n_token = n_token
        self.d_embed = d_embed
		# 边界设定
		# 向 cutoffs 数组最后加上 n_token 作为最后一个边界
        self.cutoffs = cutoffs + [n_token]
		# 添加 0
		self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

		# 所有词相同维度，共用同一个嵌入表
		if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
			# 投影维度 ≠ 嵌入维度时需要投影
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
				# 当前组的词范围
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
				# 计算嵌入维度
                d_emb_i = d_embed // (div_val ** i)
			
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

	def forward(self, inp):
        if self.div_val == 1:
			# 调用 emb_layers[0] 的 forward(), 对输入 inp 进行嵌入
            embed = self.emb_layers[0](inp)
			# 需要投影时传入 linear 改变维度
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
			# 获取参数的引用
            param = next(self.parameters())
			# 展成一维
            inp_flat = inp.view(-1)
			# 初始化嵌入矩阵
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
			# 遍历每个频率组
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
				# 通过掩码将词与对应的词频组对应起来
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
				# 为 True 的位置索引
                indices_i = mask_i.nonzero().squeeze()

				# 跳过空组
                if indices_i.numel() == 0:
                    continue
				# 减去 l_idx 将全局索引转组内索引
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
				# 分组嵌入，投影
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])
				# 通过 index_copy_ 方法填充 emb_flat 矩阵
                emb_flat.index_copy_(0, indices_i, emb_i)
			# * 是解包，将 flat 恢复成原始形状
            embed = emb_flat.view(*inp.size(), self.d_proj)
		# 将所有嵌入向量的值乘以一个缩放因子
        embed.mul_(self.emb_scale)

        return embed
```

## 2 相对编码的实现

`Transformer-XL` 中用到了原版的正弦位置编码，对应的类是 `PositionalEmbedding`，相对位置编码的实现在注意力中：

$$
\begin{align*} 
\mathbf A_{i,j}^{\mathrm{rel}} &=\underbrace{\mathbf E_{x_i}^\top \mathbf W_q^\top \mathbf W_{k,E}\mathbf E_{x_j}}_{(a)} +\underbrace{\mathbf E_{x_i}^\top \mathbf W_q^\top \mathbf W_{k,R}\textcolor{#00b9f2}{\mathbf R_{i-j}}}_{(b)} \\ &\quad+\underbrace{\textcolor{red}{u^\top} \mathbf W_{k,E}\mathbf E_{x_j}}_{(c)} +\underbrace{\textcolor{red}{v^\top} \mathbf W_{k,R} \textcolor{#00b9f2}{\mathbf R_{i-j}}}_{(d)}
\end{align*}
$$

### 2.1 RelMultiHeadAttn 类

`RelMultiHeadAttn` 是其他两个相对多头注意力 `RelPartialLearnableMultiHeadAttn` 和 `RelLearnableMultiHeadAttn` 的父类，定义类属性

```python
class RelMultiHeadAttn(nn.Module):
	def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
		super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

		# 同时生成 Q、K、V 三个矩阵
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
		# 注意力权重矩阵的 Dropout
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

		# sqrt(d_k)
        self.scale = 1 / (d_head ** 0.5)
		# post-LN 或 pre-LN
        self.pre_lnorm = pre_lnorm

	# 将绝对位置变为相对位置
	# x 是注意力矩阵 [qlen, klen, bsz, n_head]
	def _rel_shift(self, x, zero_triu=False):
		# 创建大小为 [qlen, 1, bsz, n_head] 的全 0 矩阵
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
		# 在 dim=1 将 zero_pad 拼接到 x 上，相当于给每一行查询的位置分数最前面补了一个 0
        x_padded = torch.cat([zero_pad, x], dim=1)

		# 0，1 维要交换，所以将 x.size(1) + 1
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

		# 切片舍弃第一行
		# 由于 zero_pad 的存在，使得每一行的偏移量不同
		# 原始的 i-j 相等的元素按对角线排列 --> 按列排列
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x
```

### 2.2 RelPartialLearnableMultiHeadAttn 类

```python
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
	def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
		# 位置偏置
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

	# r 是正弦位置编码，r_w_bias 是公式中的 u，r_r_bias 是公式中的 v，mems 是 上一个 seg 的记忆
	def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
	
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
		if mems is not None:
			# 拼接之前的记忆和输入 E_xj [mlen+qlen, bsz, d_model] = [klen, bsz, d_model]
            cat = torch.cat([mems, w], 0)
			# 可选的层归一化
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

			# 实现 W_kE 和 W_kR
            r_head_k = self.r_net(r)
			# 分离 Q, K, V
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
			# Q 只来自当前
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

		# [klen, bsz, n_head*d_head]
		klen = w_head_k.size(0)
		#  _len × bsz × n_head × d_head
		w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

		r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

		#### compute attention score
		# term A+C
		# w_head_q(W_q · E_xi) + r_w_bias(u)
        rw_head_q = w_head_q + r_w_bias
		'''
		einsum(equation, *operands) 
			equation: str - 描述运算的字符串
			operands: List[Tensor] - 输入张量
		输出中没有d维度，因此在d维度上做点积：
			output[i][j][b][n] = sum_d(A[i][b][n][d] * B[j][b][n][d])
		'''
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             

		# W_q·E_xi + v
        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)
```

## 3 主类
了解了嵌入和相对编码的实现后，我们最后来看 `MemTransformerLM` 类，记忆机制也在该类中实现。`Transformer-XL` 是 `decoder-only` 结构，不包含 `encoder`

### 3.1 记忆机制

```python
def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
			# 为每层创建记忆，mem[0] 是词嵌入的输出，mem[i] 是第 i 层 decoder 的输出
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

	def _update_mems(self, hids, mems, qlen, mlen):
        # 未使用 mem 不更新
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems),

        # 记忆来自之前的 seg，不需要更新梯度
        with torch.no_grad():
            new_mems = []
			# mlen：旧记忆长度, qlen: 当前段长度
			# 需要保留的记忆的索引边界
			# qlen - self.ext_len，ext_len 是留给下一个段扩展用
            end_idx = mlen + max(0, qlen - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
				# 新旧记忆拼接，[mlen+qlen, bsz, d_model]
                cat = torch.cat([mems[i], hids[i]], dim=0)
				# 保留 mlen 长度的记忆
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems
```

### 3.2 类初始化

```python
class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1):
        super().__init__()
		...
        d_embed = d_model if d_embed is None else d_embed
		# 词嵌入
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)
        self.drop = nn.Dropout(dropout)
		# 最大键长度 = 目标序列长度 + 记忆长度 + 扩展长度
        self.max_klen = tgt_len + ext_len + mem_len
        self.layers = nn.ModuleList()

		# 本文提出的 attn
		if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
			# 使用 RelLearnableDecoderLayer
            ...
        elif attn_type in [2, 3]: # absolute embeddings
	        # 使用普通的 DecoderLayer
			...

		self.sample_softmax = sample_softmax
        # 使用采样 softmax，不对全部数据进行 softmax，而是只计算 target 词 + 随机采样的一部分词的分数
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
			# 权重绑定，out_layer 和 word_emb 共享权重减少参数
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
			# 创建采样器，按对数均匀分布（平衡高频和低频词）
            self.sampler = LogUniformSampler(n_token, sample_softmax)

		else:
			# 自适应 Softmax 输出层
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)
			# 绑定输出层权重										
            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight
			# 绑定投影矩阵
            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]
		# 控制注意力掩码的形状
		self.same_length = same_length
		# 限制位置编码的最大距离
        self.clamp_len = clamp_len
        self._create_params()

	# 根据注意力类型创建不同参数
	def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
			...
        elif self.attn_type == 2: # absolute standard
            ...
        elif self.attn_type == 3: # absolute deeper SA
            ...
```

### 3.3 forward()

```python
def forward(self, data, target, *mems):
	if not mems: mems = self.init_mems()

	tgt_len = target.size(0)
	# 通过 _forward 得到隐状态和新记忆
	hidden, new_mems = self._forward(data, mems=mems)

	# 取最后 tgt_len 个预测
	pred_hid = hidden[-tgt_len:]
	if self.sample_softmax > 0 and self.training:
		assert self.tie_weight
		logit = sample_logits(self.word_emb,
			self.out_layer.bias, target, pred_hid, self.sampler)
		loss = -F.log_softmax(logit, -1)[:, :, 0]
	else:
		loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
		loss = loss.view(tgt_len, -1)

	if new_mems is None:
		return [loss]
	else:
		return [loss] + new_mems


def _forward(self, dec_inp, mems=None):

	# 输入大小的获取
    qlen, bsz = dec_inp.size()
	# 词嵌入
	word_emb = self.word_emb(dec_inp)

	mlen = mems[0].size(0) if mems is not None else 0
	# klen = 记忆长度 + 当前段长度
	klen = mlen + qlen

	# 标准因果掩码，通过 triu 只能看到当前位置及之前的内容
	if self.same_length:
		# qlen × klen 的全 1 矩阵
		all_ones = word_emb.new_ones(qlen, klen)
		# 当前段长度
		mask_len = klen - self.mem_len
		if mask_len > 0:
			# 需要遮蔽的左边列数
			mask_shift_len = qlen - mask_len
		else:
			mask_shift_len = qlen
		# 1+mlen 对角线向右移动，-mask_shift_len 向左移动
		# j - i >= 1 + mlen ，j - i <= -mask_shift_len 的位置不可见
		dec_attn_mask = (torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
	else:
		# 标准掩码只看过去
		dec_attn_mask = torch.triu(
			word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

	hids = []
	if self.attn_type == 0: # default
		# 倒序的位置序列 [klen-1, klen-2,...,0]
		pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
							   dtype=word_emb.dtype)
		# 截断，让模型专注于近期的位置关系
		if self.clamp_len > 0:
			pos_seq.clamp_(max=self.clamp_len)

		# 位置编码
		pos_emb = self.pos_emb(pos_seq)
		# 词嵌入 + dropout
		core_out = self.drop(word_emb)
		# 位置编码 + dropout
		pos_emb = self.drop(pos_emb)
		# 保存嵌入层输出
		hids.append(core_out)
		for i, layer in enumerate(self.layers):
			mems_i = None if mems is None else mems[i]
			core_out = layer(core_out, pos_emb, self.r_w_bias,
					self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
			hids.append(core_out)

	# 处理其它类型注意力
	...

	# 最后一层的 dropout
	core_out = self.drop(core_out)
	# 更新记忆
	new_mems = self._update_mems(hids, mems, mlen, qlen)
	return core_out, new_mems
```








