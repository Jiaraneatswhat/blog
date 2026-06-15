---
title: "RoBERTa 的源码 [LLM]"
published: 2026-06-15 08:00:00
category: CS
image: "./cover.png"
---

`RoBERTa (Robustly optimized BERT approach)` 是对 `BERT` 的一种改进。作者认为原版 `BERT` 存在训练不足的问题，在不改变模型架构的基础上针对预训练过程做出了改进：移除了 `SNP` 任务，改变 `BERT` 的 `MLM` 方法，对训练数据使用动态掩码策略。

模型主要定义在 `fairseq/models/roberta/model.py` 文件中，核心类和相关函数如下：

>fairseq/models/roberta/model.py
>├── 导入依赖
>├── RobertaEncoder 类
>├── RobertaModel 类
>├── RobertaClassificationHead 类
>└── 注册模型架构

## 1 MLM 的重建

`RobertaLMHead` 用于 `token` 级预训练任务: `RoBERTa` 将 `BERT` 的随机掩码更改为动态掩码，通过一个两层的 `FFN` 预测被掩码的 `token`:

```python
class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
		# 全连接层
        self.dense = nn.Linear(embed_dim, embed_dim)
		# 获取参数中配置的激活函数
        self.activation_fn = utils.get_activation_fn(activation_fn)
		# 层归一化
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
		# 只对掩码位置进行计算
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
```

## 2 分类头的实现

`RobertaClassificationHead` 用于序列级分类任务，使用 `[CLS]` 向量进行分类：

```python
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0, # 量化噪声比例
        qn_block_size=8, # 量化块大小
        do_spectral_norm=False, # 谱归一化针对的是矩阵的范数
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
		# 向前传播时向权重注入模拟量化噪声，让模型在训练时就适应低精度推理
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, features, **kwargs):
		# x 的形状是 [batch_size, seq_len, hidden_dim]，取每个 seq 的首个元素即 [CLS]
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
```

## 3 encoder 的实现

```python
class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

	# 调用了父类 FairseqEncoder 的构造函数，并把词表传进去
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
	
        # 从架构模板中补充未指定的参数
        base_architecture(args)
        self.args = args

		# 设定只保留 encoder 的哪几层，用于剪枝或其他实验
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

		# 定义嵌入层
        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

		# 创建 encoder
        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

		# 创建 lm 头
        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)
		
	# 使用 TransformerEncoder 原版 encoder
    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False, # 是否只返回隐藏状态（不经过 LM Head）
        return_all_hiddens=False,
        masked_tokens=None, # 哪些位置被掩码（用于 MLM 训练时的高效计算）
        **unused,
    ):
		# 特征提取
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
			# 通过 RobertaLMHead 输出
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
		# 输出维度是 [TBC]
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
		# 将 Fairseq 内部的 T x B x C 格式转变为 B x T x C
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions
```

## 4 模型定义

```python
class RobertaModel(FairseqEncoderModel):
	def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
		# 动态注册下游任务的分类头
        self.classification_heads = nn.ModuleDict()

	def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        encoder = RobertaEncoder(args, task.source_dictionary)

        return cls(args, encoder)

	def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

		# 从 ModuleDict 中取出对应的分类头（RobertaClassificationHead），应用到编码器输出上
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra
```

## 5 动态掩码的实现

`fairseq/data/mask_tokens_dataset.py` 文件下：

```python
class MaskTokensDataset(BaseWrapperDataset):
	def __init__(
	        self,
			# 封装 torch 的 Dataset
	        dataset: torch.utils.data.Dataset,
			# 默认 15% 的概率
			mask_prob: float = 0.15,
			mask_multiple_length = 1,
	        ...
	    ):

		# 掩码概率必须在 0，1之间
		assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
		# random_token_prob + leave_unmasked_prob 不能超过 1
		# 真正替换成 [MASK] 的概率 = 1 - random_token_prob - leave_unmasked_prob
        assert random_token_prob + leave_unmasked_prob <= 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0

		if random_token_prob > 0.0:
		    if freq_weighted_replacement:
				# 高频词更容易被抽到
		        weights = np.array(self.vocab.count)  # 按词频加权
		    else:
				# 否则均匀分布
		        weights = np.ones(len(self.vocab))
		    weights[: self.vocab.nspecial] = 0        # 特殊 token 权重为 0，特殊 token 永远不会被抽到
		    self.weights = weights / weights.sum()    # 归一化

	def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

	def __getitem_cached__(self, seed: int, epoch: int, index: int):
		# 用 (seed, epoch, index) 生成唯一随机种子，用于动态掩码
        seed = int(hash((seed, epoch, index)) % 1e6)
		# 随机数生成器
        rng = np.random.default_rng(seed)
        item = self.dataset[index]
        sz = len(item)

        # decide elements to mask
		# 创建填充相同值的数组
        mask = np.full(sz, False)
        num_mask = int(
            # add a random number for probabilistic rounding
			# 计算有多少个 token 需要掩码
            self.mask_prob * sz / float(self.mask_multiple_length)
            + rng.random()
        )

        # 连续片段掩码
		# 不放回选取 num_mask 个位置作为 "锚点"
        mask_idc = rng.choice(sz, num_mask, replace=False)
		# mask_stdev 是使用多重掩码时掩码分布的 std，多重掩码指每个锚点扩展为多个连续 token 的 Span Masking
        if self.mask_stdev > 0.0:
			# 每个锚点的 Span 长度从正态分布中随机采样
            lengths = rng.normal(
                self.mask_multiple_length, self.mask_stdev, size=num_mask
            )
			# 四舍五入取 ≥0 的
            lengths = [max(0, int(round(x))) for x in lengths]
			# 锚点及其对应长度
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ],
                dtype=np.int64,
            )
        else:
			# 固定长度的 span 长度取 mask_multiple_length
            mask_idc = np.concatenate(
                [mask_idc + i for i in range(self.mask_multiple_length)]
            )
		# Span 可能超出序列末尾，需要裁剪
        mask_idc = mask_idc[mask_idc < len(mask)]

		# Target 通道的输出逻辑：self.return_masked_tokens 时只返回被掩码位置的原词，其余位置全部填充为 pad_idx
		# 一个 MaskTokensDataset 经过 apply_mask 方法后，会同时产出两个 Dataset，分别提供训练所需的输入和标签，即 source 和 target
        if self.return_masked_tokens:
            # exit early if we're just returning the masked tokens
            # (i.e., the targets for masked LM training)
            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)
            new_item = np.full(len(mask), self.pad_idx)
            new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
            return torch.from_numpy(new_item)

        # 替换策略
		# 默认 rand_or_unmask_prob = 0.1 + 0.1 = 0.2，选中 token 的 20% 不变成 [MASK] 而是进一步处理
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
			# 从已经被掩码的位置中，随机抽取 20% 做特殊处理
            rand_or_unmask = mask & (rng.random(sz) < rand_or_unmask_prob)
			# 不随机替换成其他词 90-0-10
            if self.random_token_prob == 0.0:
                unmask = rand_or_unmask
                rand_mask = None
			# 全部随机替换 80-20-0
            elif self.leave_unmasked_prob == 0.0:
                unmask = None
                rand_mask = rand_or_unmask
			# 默认的 80-10-10
            else: 
				# 0.1 / 0.2 = 0.5
				# 特殊处理的词中一半恢复原词一半替换为随机词
                unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                decision = rng.random(sz) < unmask_prob
                unmask = rand_or_unmask & decision
                rand_mask = rand_or_unmask & (~decision)
        else:
			# 全部替换为 [MASK]
            unmask = rand_mask = None

        if unmask is not None:
            mask = mask ^ unmask

        if self.mask_whole_words is not None:
            mask = np.repeat(mask, word_lens)

		# 拷贝原始序列
        new_item = np.copy(item)
		# 替换成 [MASK]
        new_item[mask] = self.mask_idx
		# 替换成随机词
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                if self.mask_whole_words is not None:
                    rand_mask = np.repeat(rand_mask, word_lens)
                    num_rand = rand_mask.sum()

                new_item[rand_mask] = rng.choice(
                    len(self.vocab),
                    num_rand,
                    p=self.weights,
                )

        return torch.from_numpy(new_item)
```