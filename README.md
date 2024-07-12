# CINOSUM

CINOSUM 是一个针对中国多民族语言文本摘要的开源 Python 库。它旨在为研究人员和开发人员提供一个简单、高效、易于使用的工具，用以处理多元化的中国语言资源，并生成准确的摘要。CINOSUM 支持包括汉语、藏语、维吾尔语等在内的多种民族语言。

## 特点

- 支持多种中国民族语言。
- 利用先进的自然语言处理技术生成摘要。
- 灵活的接口，易于集成到现有的 Python 项目中。
- 高效的处理速度，能够快速处理大量文本。

## 安装

你可以使用 pip 命令轻松安装 CINOSUM：

```bash
pip install CINOSUM
```

## 快速上手
下面是如何使用 CINOSUM 生成一个简单的文本摘要的例子：

```python
from CINOSUM.models.CINOSUM import CINOSUM

text = ["这里是需要生成摘要的文本内容列表。"]
model = CINOSUM()

#use in chinese 
#获得两句话作为摘要
Extractive = model.Extractive(text,batch_size=2) # or Extractive = model.Extractive(text, language='zh', batch_size=2)

#use in Tibetan
#Extractive = model.Extractive(text, language='bo', batch_size=2)
#use in Uyghur
#Extractive = model.Extractive(text, language='ug', batch_size=2)
```
## 模型下载地址
| size | type         | link                |
|------|--------------|---------------------|
| 450M | transformers | [google网盘](https://drive.google.com/file/d/1bQ_uMg0LnuFJGYU5ye0obRgPTkqco4Br/view?usp=sharing) |
| 420M | Classifier   | [即将发表](https://www.runoob.com) |

之后您可以这样使用模型
```python
from CINOSUM.models.CINOSUM import CINOSUM
device = 'cpu' # or gpu(cuda)
model = CINOSUM(model_path=your_model_path, device=device)
```
## 支持的语言
目前，CINOSUM 支持以下民族语言的摘要生成：

- 汉语（zh）
- 藏语（bo）
- 维吾尔语（ug）

- ...更多语言支持即将到来！

## 贡献
我们欢迎任何形式的贡献，无论是新语言的支持、bug 修复、功能增强或是文档改进。\
\
请遵循以下步骤：

- Fork 项目到你的 GitHub 账户下。
- 创建你自己的分支（git checkout -b feature-fooBar）。
- 提交你的改变（git commit -am 'Add some fooBar'）。
- 推送到分支（git push origin feature-fooBar）。
- 创建一个新的 Pull Request。

我们提供了三种语言最简单的分句方法在`generate.py`文件中，如果您有更好的分句方法，欢迎提交PR
## 许可证
本项目采用 Apache 许可证 - 详见 LICENSE 文件。

## 支持
如果在使用 CINOSUM 时遇到任何问题，或者有任何建议，欢迎通过 GitHub 的 issues 来提交。