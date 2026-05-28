---
title: "LaTeX 入门 [LaTeX]"
published: 2026-05-25 08:00:00
category: CS
image: "./cover.png"
description: "Overleaf 中的 LaTeX 入门文档"
---


### The preamble of a document

```tex
\documentclass{article}
\begin{document}
First document. This is a simple example, with no 
extra parameters or packages included.
\end{document}
```

`\begin{document}` 前的称为导言区，可以配置文档的一些属性：
- 定义文档类型
- 指定具体设置，例如文档使用的语言
- 加载你想使用的宏包
- 其他各类格式配置

一个简单的导言区：

```tex
\documentclass[12pt, letterpaper]{article} \usepackage{graphicx}
```

- `\documentclass[12pt, letterpaper]{article}` 定义了文档的整体类型，额外参数用 `,` 分隔，并放在 `[]` 中，用于配置 `article` 的样式

- `\usepackage{graphicx}` 是加载外部宏包的一个示例，用于扩展 $\mathrm{\LaTeX}$ 的功能

### Including title, author and date information

在文档中添加标题、作者与日期，只需在导言区（而非正文区域）新增三行代码：

```tex
\documentclass[12pt, letterpaper]{article}
% 设置文档标题
\title{My first LaTeX document}
% 填写作者姓名，还可在大括号内搭配 \thanks 命令
\author{Hubert Farnsworth\thanks{Funded by the Overleaf team.}}
% 可手动填写日期；也可使用 \today 命令，每次编译都会自动生成当日日期
\date{August 2022}
```

在导言区设置好标题，作者，日期后，通过 `\maketitle` 在文档中进行输出

```tex
\begin{document}
\maketitle
We have now added a title, author and date to our first \LaTeX{} document!
\end{document}
```

### Bold, italics and underlining
- **加粗**：通过 `\textbf{...}` 实现加粗
- *斜体*：通过 `\textit{...}` 实现斜体
- <u>下划线</u>：通过 `\underline{...}` 实现下划线

还有一个实用命令 `\emph{argument}`，其效果随使用场景变化。常规正文里，该命令会将文字设为斜体；若本身处于斜体文本中，效果则会反转:

```tex
Some of the greatest \emph{discoveries} in science 
were made by accident.

\textit{Some of the greatest \emph{discoveries} 
in science were made by accident.}

\textbf{Some of the greatest \emph{discoveries} 
in science were made by accident.}
```

### Adding images

插入图片的功能需要导入 `graphicx` 宏包，它提供两个关键命令：`\includegraphics{...}` 插入项目中的图片，`\graphicspath{...}` 指定图片路径

`\includegraphics{...}` 命令中可以填写带后缀的完整文件名，但规范写法是省略后缀，程序会自动检索所有兼容图片格式

### Captions, labels and references

借助 `figure` 环境，可以为图片添加图题、设置标签并交叉引用：

```tex
\documentclass{article}
\usepackage{graphicx}
\graphicspath{{images/}}
 
\begin{document}

\begin{figure}[h]
    \centering
	% \textwidth 指 "当前页面正文的总宽度"
    \includegraphics[width=0.75\textwidth]{mesh}
	% \caption 根据当前图片顺序自动生成 'Figure n:' 拼接在 cation 前 
    \caption{A nice plot.}
	% 给图片加上标签，与 \ref{} 配合实现引用
    \label{fig:mesh1}
\end{figure}
 
As you can see in figure \ref{fig:mesh1}, the function grows near the origin. This example is on page \pageref{fig:mesh1}.

\end{document}
```

### Creating lists in LaTeX

可借助**环境命令**创建各类列表，环境用于封装实现特定排版效果的代码。环境以 `\begin{env-name}` 开头、`\end{env-name}` 结尾，`常见环境有figure`、`tabular`，以及两种列表：无序列表 `itemize`、有序列表 `enumerate`

- 无序列表

```tex
\documentclass{article}
\begin{document}
\begin{itemize}
  \item The individual entries are indicated with a black dot, a so-called bullet.
  \item The text in the entries may be of any length.
\end{itemize}
\end{document}
```

- 有序列表：会自动从 $1$ 开始生成数字序号

```tex
\documentclass{article}
\begin{document}
\begin{enumerate}
  \item This is the first entry in our list.
  \item The list numbers increase with each entry we add.
\end{enumerate}
\end{document}
```

### Adding math to LaTeX

- 行内公式：通过 `$$` 或 `\begin{math}...\end{math}` 实现

- 独立排版的公式可设置带编号或无编号：

```tex
\documentclass[12pt, letterpaper]{article}
\begin{document}
% 居中不带编号行间公式
The mass-energy equivalence is described by the famous equation
\[ E=mc^2 \] discovered in 1905 by Albert Einstein. 

In natural units ($c = 1$), the formula expresses the identity
% 居中带编号行间公式
\begin{equation}
E=m
\end{equation}
\end{document}
```

### Basic document structure

#### Abstracts

通过 `abstract` 环境来设置摘要：

```tex
\documentclass{article}
\begin{document}
% abstract 环境可以标记文本为摘要模块，成为目录、文档编译识别层级
\begin{abstract}
This is a simple paragraph at the beginning of the 
document. A brief introduction about the main subject.
\end{abstract}
\end{document}
```

#### Paragraphs and new lines

```tex
\documentclass{article}
\begin{document}

\begin{abstract}
This is a simple paragraph at the beginning of the 
document. A brief introduction about the main subject.
\end{abstract}

After our abstract we can begin the first paragraph, then press ``enter'' twice to start the second one.
% 中间空一行形成新段落
This line will start a second paragraph.
% \\ 会直接换行但不会生成新段落
I will start the third paragraph and then add \\ a manual line break which causes this text to start on a new line but remains part of the same paragraph. Alternatively, I can use the \verb|\newline|\newline command to start a new line, which is also part of the same paragraph.
\end{document}
```

这样渲染出来的段和段之间没有空行，要实现空行在开头引入 `parskip` 包

#### Chapters and sections

```tex
\documentclass{book}
\begin{document}

% 自动生成 Chapter 1 \n {chapter_name}
\chapter{chapter_name}

% 根据 \chapter 的顺序和 \section 的顺序加上数字前缀
\section{Introduction}

This is the first section.

\section{Second Section}

...


% 根据所在 \section 生成三级标题
\subsection{First Subsection}
Praesent imperdietmi nec ante. Donec ullamcorper, felis non sodales...

% 在各级标题后加上 * 可以取消自动编号
\section*{Unnumbered Section}
Lorem ipsum dolor sit amet, consectetuer adipiscing elit.  
Etiam lobortis facilisissem...
\end{document}
```

#### Creating tables
- 一个简单的表格

```tex
% 整体居中
\begin{center}
% 默认方式使用 tabular 环境，{c c c} 参数表示有三个居中的列
% 也可使用 r, l 实现左居中和右居中
\begin{tabular}{c c c}
 cell1 & cell2 & cell3 \\ % 列之间使用 &, 行之间使用 \\
 cell4 & cell5 & cell6 \\  
 cell7 & cell8 & cell9    
\end{tabular}
\end{center}
```

- 给表格加边框：

```tex
\begin{center}
 % 在列参数中使用 | 添加竖线
\begin{tabular}{|c|c|c|} 
 % 通过 \hline 添加横线
 \hline
 cell1 & cell2 & cell3 \\ 
 cell4 & cell5 & cell6 \\ 
 cell7 & cell8 & cell9 \\ 
 \hline
\end{tabular}
\end{center}

% 也可以添加多个横线或竖线：
\begin{center}
 \begin{tabular}{||c c c c||} 
 \hline
 % [0.5ex] 在行末增加 0.5ex 的高度间距
 Col1 & Col2 & Col2 & Col3 \\ [0.5ex] 
 \hline\hline
 1 & 6 & 87837 & 787 \\ 
 \hline
 2 & 7 & 78 & 5415 \\
 \hline
 3 & 545 & 778 & 7507 \\
 \hline
 4 & 545 & 18744 & 7560 \\
 \hline
 5 & 88 & 788 & 6344 \\ [1ex] 
 \hline
\end{tabular}
\end{center}
```

- 给表格加注释标签等

```tex
% \ref 同图片的 label 引用
Table \ref{table:data} shows how to add a table caption and reference a table.
\begin{table}[h!] % 加 ！强制把表格放在当前位置
\centering
\begin{tabular}{||c c c c||} 
 \hline
 Col1 & Col2 & Col2 & Col3 \\ [0.5ex] 
 \hline\hline
 1 & 6 & 87837 & 787 \\ 
 2 & 7 & 78 & 5415 \\
 3 & 545 & 778 & 7507 \\
 4 & 545 & 18744 & 7560 \\
 5 & 88 & 788 & 6344 \\ [1ex] 
 \hline
\end{tabular}
% 自动拼接 Table n: 
\caption{Table to test captions and labels.}
\label{table:data}
\end{table}
```

#### Adding a Table of Contents

```tex
\documentclass{article}
\title{Sections and Chapters}
\author{Gubert Farnsworth}
\date{August 2022}
\begin{document}
  
\maketitle
% 自动生成目录
\tableofcontents
```
