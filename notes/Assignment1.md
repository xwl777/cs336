* problem1
	* (a)'\x00'
	* (b) 它是一个长度为1的有效不可见字符串
	* (c)一个显示'\x00'，一个则是不可见字符
* problem2
	* (a)ASCII 字符在 UTF-8 中每个字符只占 1 byte；UTF-16 通常占 2 bytes；UTF-32 固定 4 bytes。很多训练语料包含大量英文、数字、标点、代码、空格等，因此 UTF-8 会显著减少基础 byte 序列长度。英文文本在 UTF-16/UTF-32 中会包含很多 00，这些零字节不是有意义的文本结构，但会强烈影响 BPE 的 merge 统计。模型和 tokenizer 可能会浪费很多词表容量去学习编码格式的 artifacts，而不是语言本身的规律。
	* (b) 它把每个 byte 单独当成一个完整的 UTF-8 字符来解码
	*  b"\x80\x80" 原因：在 UTF-8 中，0x80 到 0xBF 是 continuation byte，只能出现在多字节字符的后续位置，不能作为一个字符的起始 byte。
* BPE分词器
	* regex库
		* re.compile() 会把正则表达式字符串预编译成一个正则表达式对象，类型通常是 re.Pattern。之后可以反复用这个对象做匹配、搜索、替换等操作。
			* pat.match(text)      # 从字符串开头匹配
			* pat.search(text)     # 在字符串中查找第一个匹配
			* pat.findall(text)    # 找出所有匹配
			* pat.finditer(text)   # 返回所有匹配的迭代器
			* pat.sub(repl, text)  # 替换匹配内容
		* re.escape() 用来把字符串 s 里的正则特殊字符转义，让它们在正则里按“普通字符”匹配。原理上，re.escape() 会扫描输入字符串，把在正则表达式中可能有特殊意义的字符前面加上反斜杠。
	* 优化思路
		1. 最开始的瓶颈在于update_cnt，原因是每一次merge都要扫描所有的word。优化思路是维护一个pair->word的反向索引，记录所有受影响的word。
		 2. 之后的瓶颈在于max_pair，要扫描所有pair进行比较。可以使用堆优化，但是实现较麻烦，暂时搁置。
		 3. 发现count_word调用的大量的word2byte，因为每扫描一个word都调用一次。然而word有大量重复，所有重复的word调用一次即可

* 