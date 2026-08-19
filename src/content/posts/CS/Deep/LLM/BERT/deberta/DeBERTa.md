---
title: DeBERTa 的一个简单实现 [LLM]
published: 2026-08-19 09:00:00
category: CS
image: ./cover.png
---

`DeBERTa` 是微软 `ICLR2021` 提出的一种 `encoder-only` 预训练语言模型，主要的核心创新在于解耦注意力和增强掩码解码器

代码仓库在 `microsoft/DeBERTa/tree/master/DeBERTa` 下，目录结构如下：

> deberta
├── __init__.py 
├── deberta.py # DeBERTa主模型
├── disentangled_attention.py # 解耦注意力 
├── da_util.py # 相对位置矩阵
├── configuration.py 
├── tokenization.py # V1: GPT2‑BPE；V2: SentencePiece分词封装 
├── ops.py # 自定义算子, XSoftmax, StableDropout, MaskedLayerNorm 
└── utils.py

### 1 注意力解耦
回顾之前论文中解耦注意力的公式：

$$
\begin{align*} 
\mathbf Q_c = \mathbf H \mathbf W_{q,c},\; &\mathbf K_c = \mathbf H \mathbf W_{k,c},\; \mathbf V_c = \mathbf H \mathbf W_{v,c},\; \mathbf Q_r = \mathbf P \mathbf W_{q,r},\; \mathbf K_r = \mathbf P \mathbf W_{k,r} \\[4pt] &\tilde{A}_{i,j} = \underbrace{\mathbf Q_i^c {\mathbf K_j^c}^\top}_{(\text{a}) \text{ content-to-content}} + \underbrace{\mathbf Q_i^c {\mathbf K_{\delta(i,j)}^r}^\top}_{(\text{b}) \text{ content-to-position}} + \underbrace{\mathbf K_j^c {\mathbf Q_{\delta(j,i)}^r}^\top}_{(\text{c}) \text{ position-to-content}}  \\[4pt] &\mathbf H_o = \mathrm{softmax}\left( \frac{\tilde{A}}{\sqrt{3d}} \right) \mathbf V_c 
\end{align*} 
$$

其中涉及到三项 `c2c`, `c2p` 和 `p2c`

对应的实现在 disentangled_attention.py 中：

#### 1.1 DisentangledSelfAttention 类
##### `__init__()` 
```python
class DisentangledSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.num_attention_heads = config.num_attention_heads
		# 每个 head 的维度
		_attention_head_size = int(config.hidden_size / config.num_attention_heads)
		# 检查配置里是否写了 head 的维度
		self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		
		# Q_c, K_c, V_c: 将输入从 hidden_size 映射到 all_head_size, 后面再分头，这样只需要一个线性层
		# 类似 transfoemer 中的 self.w_q = nn.Linear(d_model, d_model, bias=False)
		self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
		self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
		self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
		
		# 是否在内容和位置之间共享投影权重, 默认 False Q_r, K_r 有自己的 W
		self.share_att_key = getattr(config, 'share_att_key', False)
		
		# c2p, p2c (c2c 不包含在位置选项内)
		self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'c2p').lower().split('|')]
		
		# 是否使用相对位置 attn
		self.relative_attention = getattr(config, 'relative_attention', False)
		
		if self.relative_attention:
		# pos 桶数, 根据距离存储到不同的桶中, 节省空间
			self.position_buckets = getattr(config, 'position_buckets', -1)
			self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
			if self.max_relative_positions < 1:
			self.max_relative_positions = config.max_position_embeddings
			# pos 嵌入表大小, 分桶时取桶数, 因为距离只会是这些桶中的一个
			self.pos_ebd_size = self.max_relative_positions
			if self.position_buckets > 0:
			self.pos_ebd_size = self.position_buckets
			# For backward compitable
			self.pos_dropout = StableDropout(config.hidden_dropout_prob)
			if (not self.share_att_key):
			if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
			# c2p 项 计算 K_r 时的 W_{k,r}
			self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
			if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
			# W_{q,r}
			self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)
			self.dropout = StableDropout(config.attention_probs_dropout_prob)
			# 新旧版本模型权重格式转换
			self._register_load_state_dict_pre_hook(self._pre_load_hook)
```

##### `forward()`
```python
def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
	# 若没有查询 (例如自注意力)
	if query_states is None:
		query_states = hidden_states

	# self.x_proj() 调了 forward(), 得到了 [batch, seq, all_head_size] 的 x_proj
	# 在最后一维上进行分头: [batch * heads, seq, head_size]
	query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads).float()
	key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads).float()
	value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

	rel_att = None
	# 分母上的缩放项
	scale_factor = 1
	if 'c2p' in self.pos_att_type:
		scale_factor += 1
	if 'p2c' in self.pos_att_type:
	scale_factor += 1
	if 'p2p' in self.pos_att_type:
		scale_factor += 1

	# 在单头 attn 中分母 d 对应的是隐维度, 这里的 head_size * scale = qk/√3d 对应隐维度
	scale = 1/math.sqrt(query_layer.size(-1) * scale_factor)

	attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) * scale)

	# 计算解耦注意力分数
	if self.relative_attention:
		rel_embeddings = self.pos_dropout(rel_embeddings)
		rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

	# [batch×h, q, k]
	if rel_att is not None:
		attention_scores = (attention_scores + rel_att)
	attention_scores = (attention_scores - attention_scores.max(dim=-1, keepdim=True).values.detach()).to(hidden_states)

	# batch 和 heads 分开
	attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

	_attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
	attention_probs = self.dropout(_attention_probs)
	context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
	context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
	new_context_layer_shape = context_layer.size()[:-2] + (-1,)
	context_layer = context_layer.view(*new_context_layer_shape)

	return {
		'hidden_states': context_layer,
		'attention_probs': _attention_probs,
		'attention_logits': attention_scores
		}
```

##### `transpose_for_scores()`
```python
# 分头
def transpose_for_scores(self, x, attention_heads):
	# x.size()[:-1] 是除最后一维前的所有维度, 拼接后面的维度
	# 最后一维拆分成 attention_heads * -1 (head_size 的大小让 torch 自己算)
	# [batch, seq, heads, head_size]
	new_x_shape = x.size()[:-1] + (attention_heads, -1)
	
	# 改变 x 的形状, 通过 * 进行解包
	x = x.view(*new_x_shape)

	# permute 交换 seq 和 heads
	# contiguous() 重新排列内存, 因为 permute 后的内存不是连续的
	# 再通过 view 将 batch 和 heads 合并成一个维度: [batch * heads, seq, head_size]
	return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
```

##### `disentangled_attention_bias()`
```python
# 计算解耦注意力权重
def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
	
	if relative_pos is None:
	# seq_len
	q = query_layer.size(-2)
	# q size, k size, bkt size
	# 生成一个矩阵记录 token 间的距离
	relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions, device=query_layer.device)

	if relative_pos.dim()==2:
		relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
	# [1, q, k] -> [batch, 1, q, k]
	elif relative_pos.dim()==3:
		relative_pos = relative_pos.unsqueeze(1)
	elif relative_pos.dim()!=4:
		raise ValueError(...)
  
	att_span = self.pos_ebd_size
	relative_pos = relative_pos.long().to(query_layer.device)

	# bert.py 中 self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size), 大小是 [pos_ebd_size, hidden_size]
	# att_span = self.pos_ebd_size 带入: 切片 [0: 2*pos_ebd_size] 添加维度 -> [1, 2*pos_ebd_size, hidden_size]
	rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span: self.pos_ebd_size + att_span, :].unsqueeze(0) 

	if self.share_att_key:
		...

	else:
	# 把之前的 proj 投影成 K^r 和 Q^r
		if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
			# [1, 2*pos_ebd_size, hidden_size] 的 embedding 经过 pos_key_proj 投影成 [1, 2*pos_ebd_size, all_head_size]
			# 分头 -> [1*heads, 2*pos_ebd_size, head_size]
			# repeat(*sizes, 每个维度重复几次)
			# size(0) 是 batch * heads, // heads 得到 batch
			# rel_embeddings 是共享的, 为了和 query_layer [batch * heads, seq, head_size] 做 bmm 要复制 batch 份
			# [B, m, n] @ [B, n, p] = [B, m, p]
			pos_key_layer = 
				self.transpose_for_scores(
					self.pos_key_proj(rel_embeddings), 
					self.num_attention_heads)\
				# heads 重复 batch 次
				.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)

		# 计算 Q^r
		if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
			pos_query_layer = 
				self.transpose_for_scores(
					self.pos_query_proj(rel_embeddings), 
					self.num_attention_heads)\
				.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) 
				
	score = 0
	# content->position
	if 'c2p' in self.pos_att_type:
		scale = 1/math.sqrt(pos_key_layer.size(-1)*scale_factor)
	# query_layer: [batch×heads, q, head_size], pos_key_layer: [batch×heads, 2*pos_ebd, head_size] 后者转置才能做乘法
		c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2).to(query_layer)*scale)

		# 把桶编号转换成数组索引
		# expand() 扩展到 [batch×heads, q, 2 * pos_ebd]
		c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1).squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])

		# 2 * pos_ebd 是所有距离的分数, 根据 gather 取出对应项
		c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
		score += c2p_att

	# position->content
	if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
		scale = 1/math.sqrt(pos_query_layer.size(-1)*scale_factor)

	if 'p2c' in self.pos_att_type:
		# 为了复用 c2p_pos, 将公式中的 Q_c · K_p^T 进行了转置
		# 转置后 2*pos 的位置变了, 需要让 dim=-2
		p2c_att = torch.bmm(pos_query_layer.to(key_layer)*scale, key_layer.transpose(-1, -2))
		p2c_att = torch.gather(p2c_att, dim=-2, index=c2p_pos)
		score += p2c_att

	return score
```

#### 1.2 da_utils.py
##### `build_relative_position()`

```python
# 计算 token 间的相对距离
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
	q_ids = torch.arange(0, query_size)
	k_ids = torch.arange(0, key_size)
	if device is not None:
	    q_ids = q_ids.to(device)
	    k_ids = k_ids.to(device)
	
	# q_ids 转列向量, k_ids 转行向量 ((n,) -> (n,1)), 利用广播运算快速算出 token 间所有的距离
	rel_pos_ids = q_ids.view(-1,1) - k_ids.view(1,-1)

	if bucket_size > 0 and max_position > 0:
		# 分桶
		rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)

	# 只关注需要查询的行
	rel_pos_ids = rel_pos_ids[:query_size, :]
	rel_pos_ids = rel_pos_ids.unsqueeze(0)
	return rel_pos_ids
```

##### `make_log_bucket_position()`

```python
def make_log_bucket_position(relative_pos, bucket_size, max_position):
	# 通过 clamp + max_position 后将距离转为正值
	relative_pos = torch.clamp(relative_pos,-max_position+1, max_position-1) + max_position
	# [pos1, pos2, ...]
	bucket_dict = make_log_bucket_dict(bucket_size, max_position, relative_pos.device)
	for d in range(relative_pos.dim()-1):
	    # 插入一个 1 的新维度, 方便和 bkt_pos 广播
	    bucket_dict = bucket_dict.unsqueeze(0)
	    # gather () 根据索引从 tensor 中提取元素
	    # 取 relative_pos.size() 的前 3 个维度, 与 bucket_dict 的最后一个维度拼接
	    # 通过 index=relative_pos 进行索引
	    bucket_pos = torch.gather(bucket_dict.expand(list(relative_pos.size())[:-1] + [bucket_dict.size(-1)]), index=relative_pos.long(), dim=-1)
	return bucket_pos

def make_log_bucket_dict(bucket_size, max_position, device=None):
	relative_pos = torch.arange(-max_position, max_position, device=device)
	# 对数压缩会丢失符号，这里先存起来
	sign = torch.sign(relative_pos)
	# 线性区和对数区的分界点
	mid = bucket_size // 2
	# 距离在 (-mid, mid) 内的线性处理
	abs_pos = torch.where((relative_pos<mid) & (relative_pos > -mid), torch.tensor(mid-1).to(relative_pos), torch.abs(relative_pos))
	# mid, ..., max_pos - 1 转换为 mid, ..., 2 * mid - 1
	log_pos = torch.ceil(torch.log(abs_pos/mid)/math.log((max_position-1)/mid) * (mid-1)) + mid
	# 合并线性区和对数区
	# where 是三元运算符
	# mid 范围内的取原始的 relative_pos, 否则给 log 距离乘上之前的 sign
	bucket_pos = torch.where(abs_pos<=mid, relative_pos, (log_pos*sign).to(relative_pos)).to(torch.long)
	return bucket_pos
```

### 2 增强掩码 decoder

`DeBERTa` 在全部 `Transformer` 层运算完成后、`token` 预测的 `softmax` 层之前引入绝对位置信息:

```python
# masked_language_model.py 中:
class EnhancedMaskDecoder(torch.nn.Module):
  def __init__(self, config, vocab_size):
    super().__init__()
    self.config = config
    self.position_biased_input = getattr(config, 'position_biased_input', True)
    self.lm_head = BertLMPredictionHead(config, vocab_size)

  def forward(self, ctx_layers, ebd_weight, target_ids, input_ids, input_mask, z_states, attention_mask, encoder, relative_pos=None):
	# 获取增强的上下文表示
    mlm_ctx_layers = self.emd_context_layer(ctx_layers, z_states, attention_mask, encoder, target_ids, input_ids, input_mask, relative_pos=relative_pos)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    lm_loss = torch.tensor(0).to(ctx_layers[-1])
    arlm_loss = torch.tensor(0).to(ctx_layers[-1])
	# 最后一层输出
    ctx_layer = mlm_ctx_layers[-1]
    lm_logits = self.lm_head(ctx_layer, ebd_weight).float()
    lm_logits = lm_logits.view(-1, lm_logits.size(-1))
    lm_labels = target_ids.view(-1)
    label_index = (target_ids.view(-1)>0).nonzero().view(-1)
    lm_labels = lm_labels.index_select(0, label_index)
    lm_loss = loss_fct(lm_logits, lm_labels.long())
    return lm_logits, lm_labels, lm_loss

  def emd_context_layer(self, encoder_layers, z_states, attention_mask, encoder, target_ids, input_ids, input_mask, relative_pos=None):
    if attention_mask.dim()<=2:
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		att_mask = extended_attention_mask.byte()
		attention_mask = att_mask*att_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim()==3:
		attention_mask = attention_mask.unsqueeze(1)
		target_mask = target_ids>0
	# 取倒 2 层不含 MLM 信息的表示
    hidden_states = encoder_layers[-2]
	# position_biased_input 为 False 时使用 emd_context
    if not self.position_biased_input: 
		# encoder 的最后一层
		layers = [encoder.layer[-1] for _ in range(2)]
		# 融合所有的隐状态作为 query
		z_states += hidden_states
		query_states = z_states
		query_mask = attention_mask
		outputs = []
		rel_embeddings = encoder.get_rel_embedding()

	for layer in layers:
        output = layer(hidden_states, query_mask, return_att=False, query_states = query_states, relative_pos=relative_pos, rel_embeddings = rel_embeddings)
		# 跑两次, 第一次的结果作为第二次的 query
        query_states = output
        outputs.append(query_states)
    else:
		# 否则直接取最后一层
		outputs = [encoder_layers[-1]]
    
    _mask_index = (target_ids>0).view(-1).nonzero().view(-1)
	
    def flatten_states(q_states):
		q_states = q_states.view((-1, q_states.size(-1)))
		q_states = q_states.index_select(0, _mask_index)
		return q_states
		
    return [flatten_states(q) for q in outputs]
```

### 3 模型主类

```python
# deberta.py
class DeBERTa(torch.nn.Module):
  def __init__(self, config=None, pre_trained=None):
    super().__init__()
    state = None
	# 加载预训练权重
    if pre_trained is not None:
		state, model_config = load_model_state(pre_trained)
		if config is not None and model_config is not None:
			for k in config.__dict__:
				if k not in ['hidden_size',
					'intermediate_size',
					'num_attention_heads',
					'num_hidden_layers',
					'vocab_size',
					'max_position_embeddings']:
					model_config.__dict__[k] = config.__dict__[k]
		config = copy.copy(model_config)
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.config = config
    self.pre_trained = pre_trained
	# 加载已有状态
    self.apply_state(state)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_all_encoded_layers=True, position_ids = None, return_att = False):
    
	if attention_mask is None:
		attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
		token_type_ids = torch.zeros_like(input_ids)

    ebd_output = self.embeddings(input_ids.to(torch.long), token_type_ids.to(torch.long), position_ids, attention_mask)
    embedding_output = ebd_output['embeddings']
    encoder_output = self.encoder(embedding_output,
                   attention_mask,
                   output_all_encoded_layers=output_all_encoded_layers, return_att = return_att)
    encoder_output.update(ebd_output)
    return encoder_output
```

