import regex as re
from collections import defaultdict

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")# pre-tokenization的正则表达式

# 从文件读取训练文本
def read_text(input_path:str):
    with open(input_path, "r", encoding = "utf-8") as f:
        text = f.read()
    return text

# 根据特殊字符切分原始文本
def special_split(special_tokens:list[str], text:str, drop = True):
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join([re.escape(token) for token in special_tokens])

    if not drop:
        pattern = f"({pattern})" # 加（）为捕获组，在split时将会保留分隔符

    pattern = re.compile(pattern)

    chunks = pattern.split(text)
    return [c for c in chunks if c] # split可能会分割出空字符串

# 把unicode字符转化为byte元组（tupel[byte]）
def word2byte(word:str):
    encoded = list(word.encode("utf-8")) # 一个int数组
    return tuple([bytes([index]) for index in encoded])

# 初始化词表
def vocab_init(special_tokens:list[str]):
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    # print(vocab)
    return vocab

# 计算文本由正则分为多个chunk后各个chunk出现的次数
def count_word(text:str):
    cnt = defaultdict(int)
    for m in PAT.finditer(text):
        word = m.group()
        word_bytes = word2byte(word)
        if len(word_bytes) >= 2: # 这个cnt计数器时用于后续方便计算pair出现的次数，长度为1的word_bytes不产生任何pair，故可以直接忽略
            cnt[word_bytes] += 1
    return cnt

# 原始文本可能被special_tokens分为多个文本块，故需要合并得到的计数字典
def merge_cnt(cnt_lst:list[dict]):
    total_cnt = defaultdict(int)
    for cnt in cnt_lst:
        for k, v in cnt.items():
            total_cnt[k] += v

    return total_cnt

# 返回pair的计数
def count_pairs(word_cnt:dict):
    pairs_cnt = defaultdict(int)
    for word,cnt in word_cnt.items():
        for pair in zip(word[:-1],word[1:]): #将两个可迭代对象打包成元组列表
            pairs_cnt[pair] += cnt

    return pairs_cnt

# 选择计数最大的那个pair
def max_pair(pairs_cnt:dict):
    return max(pairs_cnt.items(),key = lambda x:(x[1], x[0]))[0]

# 进行pair的合并
def merge(word_bytes:tuple[bytes],merge:tuple[bytes]):
    merged_token = merge[0] + merge[1]
    new_word_bytes = list()
    i = 0
    while i < len(word_bytes):
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]:
            new_word_bytes.append(merged_token)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

# 合并pair后需要更新pair计数
def update_cnt(word_cnt:dict, pair_cnt:dict, merge_pair:tuple[bytes]):
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int, pair_cnt) # 复制原pair_cnt

    for word_bytes, cnt in word_cnt.items():
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
        if merge_pair not in old_pairs: 
            new_word_cnt[word_bytes] += cnt
            continue # 该词中没有出现合并的token则无影响

        # 否则将key更新为合并后的word_bytes
        new_word_bytes = merge(word_bytes, merge_pair)
        new_word_cnt[new_word_bytes] += cnt

        # 重新计算该词对pair_cnt的影响，将原计数全部删去后重新计数
        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair] 
        
        for pair in zip(new_word_bytes[:-1], new_word_bytes[1:]):
            new_pair_cnt[pair] += cnt

    return new_word_cnt, new_pair_cnt

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    text = read_text(input_path)
    chunks = special_split(special_tokens, text)
    print("special tokens 分割完成")
    cnt_lst = list(map(count_word, chunks))
    word_cnt = merge_cnt(cnt_lst)
    print("word 计数完成")
    pairs_cnt = count_pairs(word_cnt)
    print("配对计数完成")
    vocab = vocab_init(special_tokens)
    merges = list()
    
    print("初始化完成，开始训练")
    while(len(vocab) < vocab_size):
        if(len(vocab) % 100 == 0):
            print(f"当前进度{len(vocab)}/{vocab_size}")
        merge_pair = max_pair(pairs_cnt)
        merges.append(merge_pair)
        word_cnt, pairs_cnt = update_cnt(word_cnt, pairs_cnt, merge_pair)
        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]

    return vocab, merges

