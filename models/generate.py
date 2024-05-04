
import torch
import copy
import numpy as np
import re
from transformers import XLMRobertaModel,XLMRobertaConfig,XLMRobertaTokenizer
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from tqdm import tqdm

tokenizer = XLMRobertaTokenizer.from_pretrained("hfl/cino-large-v2")
def truncate_context_to_fit_tokenizer(context, tokenizer, max_length=512):
    # 首先对原始的context进行分词
    context_tokens = tokenizer.tokenize(context)
    # 如果分词后的长度小于或等于max_length，就直接返回原始context
    if len(context_tokens) < max_length:
        return context

    # 如果分词后的长度大于max_length，需要截断
    # 注意：这里我们不再添加空格，而是直接截取token列表
    truncated_tokens = context_tokens[:max_length]

    # 使用分词器的convert_tokens_to_string方法将token列表转换回字符串
    truncated_context = tokenizer.convert_tokens_to_string(truncated_tokens)

    #print(truncated_context)

    return truncated_context

def normal_cut_sentence(text):
    text = re.sub('([。！？\?])([^’”])',r'\1\n\2',text)#普通断句符号且后面没有引号
    text = re.sub('(\.{6})([^’”])',r'\1\n\2',text)#英文省略号且后面没有引号
    text = re.sub('(\…{2})([^’”])',r'\1\n\2',text)#中文省略号且后面没有引号
    text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])',r'\1\n\2',text)#断句号+引号且后面没有引号
    return text.split("\n")

def tokenize_zh_better(text):
    p = re.compile("“.*?”")
    list = []
    index = 0
    length = len(text)
    for i in p.finditer(text):
        temp = ''
        start = i.start()
        end = i.end()
        for j in range(index, start):
            temp += text[j]
        if temp != '':
            temp_list = normal_cut_sentence(temp)
            list += temp_list
        temp = ''
        for k in range(start, end):
            temp += text[k]
        if temp != ' ':
            list.append(temp)
        index = end
    return list

def tokenize_zh_base(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def tokenize_bo(text):
    # 这里假设藏文句子结束可以用特殊的藏文标点符号 "།" 来识别
    sentences = text.split('།')
    # 去除句子两边的空白并过滤掉空句子
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def tokenize_ug(text):
    # 这里假设维文句子结束可以用句号 "。"、问号 "؟" 和感叹号 "！" 来识别
    # 注意维吾尔文中使用的是阿拉伯问号 "؟" 而不是标准问号 "?"
    sentences = []
    sentence = ''
    for char in text:
        sentence += char
        if char in ('。', '؟', '！','.'):
            sentences.append(sentence.strip())
            sentence = ''
    # 确保最后一个句子被添加，即使它后面没有标点符号
    if sentence:
        sentences.append(sentence.strip())
    return sentences


def sentence_tokenize(text, language='zh'):
    """
    Tokenize a text into sentences for the specified language.

    Parameters:
    text (str): The text to be tokenized.
    language (str): The language code for the tokenizer. Supported languages are 'zh', 'bo', and 'ug'.

    Returns:
    list: A list of sentences.
    """
    
    # 首先检查 text 是否为字符串
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    

    # 根据传入的语言参数调用相应的分句方法
    if language == 'zh' or language == '中文':
        return tokenize_zh_base(text)
    elif language == 'bo' or language == '藏文':
        return tokenize_bo(text)
    elif language == 'ug' or language == '维文':
        return tokenize_ug(text)
    else:
        # 如果传入了一个未知的语言代码，抛出一个异常
        raise ValueError(f"Unsupported language code: {language}")



class CustomDataset(Dataset):
    def __init__(self, input_ids, segs, clss, attention_masks, mask_cls, src_str):
        self.input_ids = input_ids
        self.segs = segs
        self.clss = clss
        self.attention_masks = attention_masks
        self.mask_cls = mask_cls
        self.src_str = src_str

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'segs': self.segs[index],
            'clss': self.clss[index],
            'attention_masks': self.attention_masks[index],
            'mask_cls': self.mask_cls[index],
            'src_str': self.src_str[index]  # 这里把句子也包括进来
        }

def custom_collate_fn(batch):
    # 假设batch是一个列表，其中每个元素是一个字典，拥有相同的键
    # 例如：[{'input_ids': tensor1, 'segs': tensor2, ..., 'src_str': list1}, {...}, ...]

    # 初始化字典以收集所有数据字段
    batch_data = {
        'input_ids': [],
        'segs': [],
        'clss': [],
        'attention_masks': [],
        'mask_cls': [],
        'src_str': []
    }

    for item in batch:
        batch_data['input_ids'].append(item['input_ids'])
        batch_data['segs'].append(item['segs'])
        batch_data['clss'].append(item['clss'])
        batch_data['attention_masks'].append(item['attention_masks'])
        batch_data['mask_cls'].append(item['mask_cls'])
        # src_str 是一个句子列表，直接加入列表
        batch_data['src_str'].append(item['src_str'])

    # 对于前五个字段，我们使用torch.stack来合并列表中的tensor
    for key in ['input_ids', 'segs', 'clss', 'attention_masks', 'mask_cls']:
        batch_data[key] = torch.stack(batch_data[key], dim=0)

    # src_str 保持为列表形式，不需要stack
    # batch_data['src_str'] 本来就已经是列表了，不需要额外处理

    return batch_data

def process_text_2_test_iter(contexts, device, len_pred = 2, language='zh', batch_size=1, print_setences=False):
    # 这将存储所有文本的处理结果
    all_input_ids = []
    all_attention_masks = []
    all_segs = []
    all_clss = []
    all_mask_cls = []
    all_src_str = []
    max_length = 512

    for context in contexts:

        #尽量保留原文本
        context = truncate_context_to_fit_tokenizer(context, tokenizer, max_length = max_length)
        
        # 分句
        sentences = sentence_tokenize(context, language)
        sum_sentences_length = sum((len(sentence) for sentence in sentences))

        if sum_sentences_length>=max_length:
            context = context[:max_length-1]
            sentences = sentence_tokenize(context, language)

        #assert len(sentences)<len_pred

        # tokenizer处理，获取input_ids和attention_mask
        encoding = tokenizer(context, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # segs全零张量,CINO没有token_typeid
        segs = torch.zeros(max_length, dtype=torch.long)
        
        # clss为每个句子的起始位置
        clss = [-1] * input_ids.size(1)  # 初始化clss列表
        clss[0] = 0
        for i, sentence in enumerate(sentences):
            clss[i+1] = clss[i]+len(sentence)

            if print_setences:
                print(f'第:{i+1}个句子:{sentence}')
            
        
        clss = torch.tensor(clss)
        # mask_cls
        mask_cls = ~ (clss == -1)
        clss[clss == -1] = 0
        
        # 累积处理结果
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_segs.append(segs)
        all_clss.append(clss)
        all_mask_cls.append(mask_cls)
        all_src_str.append(sentences)

    num_samples = len(all_input_ids)
    
    # Flatten all_input_ids, all_attention_masks, and all_segs, assuming they are lists of 2D tensors
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    
    # Flatten all_clss and all_mask_cls, and make sure they match the batch dimension
    # Assuming all_clss and all_mask_cls are lists of 1D tensors
    all_segs = torch.cat([segs.unsqueeze(0) for segs in all_segs], dim=0)
    all_clss = torch.cat([cls.unsqueeze(0) for cls in all_clss], dim=0)
    all_mask_cls = torch.cat([mask.unsqueeze(0) for mask in all_mask_cls], dim=0)


    # Check that the batch dimension matches
    assert all_input_ids.size(0) == num_samples
    assert all_attention_masks.size(0) == num_samples
    assert all_segs.size(0) == num_samples
    assert all_clss.size(0) == num_samples
    assert all_mask_cls.size(0) == num_samples

    #print(f'all_input_ids:{all_input_ids.shape}')
    #print(all_src_str)
    #print(len(all_src_str))
    # 创建TensorDatasets
    dataset = CustomDataset(
        all_input_ids,
        all_segs,
        all_clss,
        all_attention_masks,
        all_mask_cls,
        all_src_str
    )
    
    # 使用DataLoader来处理batching
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False)

    
    return data_loader



def generate(model, contexts, device, batch_size = 1, len_pred = 2, language = 'zh'):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False
        
        pred_list = []
        
        test_iter = process_text_2_test_iter(contexts, device, batch_size=batch_size, len_pred=len_pred, language = language)
        model.eval()
        with torch.no_grad():
            
            for batch in tqdm(test_iter, desc="Processing", leave=True):
                
                src = batch['input_ids'].to(device)
                segs = batch['segs'].to(device)
                clss = batch['clss'].to(device)
                mask = batch['attention_masks'].to(device)
                mask_cls = batch['mask_cls'].to(device)
                src_str = batch['src_str']

                # src = batch[0].to(device)
                # segs = batch[1].to(device)
                # clss = batch[2].to(device)
                # mask = batch[3].to(device)
                # mask_cls = batch[4].to(device)
                

                
                pred = []
                              
                sent_scores, mask = model(src, segs, clss, mask, mask_cls)                           
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                selected_ids = np.argsort(-sent_scores, 1)

                for i, idx in enumerate(selected_ids):
                    _pred = []
                    if(len(src_str[i])==0):
                        continue
                    
                    for j in selected_ids[i][:len(src_str[i])]:
                        if(j>=len(src_str[i])):
                            continue

                        candidate = src_str[i][j].strip()
                        if(not _block_tri(candidate,_pred)):
                                _pred.append(candidate)
                        

                        if (len(_pred) == len_pred):
                            break

                    pred_list.append(copy.copy(_pred))

                    _pred = '<q>'.join(_pred)

                    pred.append(_pred)
                
        return pred_list

            