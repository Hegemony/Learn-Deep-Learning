import torch
import random
import zipfile

with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:  # 通过 Python 内置的 zipfile 模块实现对 zip 文件的解压
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])  # 输出前40个歌词
print('-'*100)
'''
这个数据集有6万多个字符。为了打印方便，我们把换行符替换成空格，然后仅使用前1万个字符来训练模型。
'''
'''
Python replace() 方法把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次。
语法:
replace()方法语法：
str.replace(old, new[, max])
'''
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')  # 将“换行”和“回车”符号替换为空格
corpus_chars = corpus_chars[0:10000]
print(corpus_chars)
print('-'*100)

'''
建立字符索引：
我们将每个字符映射成一个从0开始的连续整数，又称索引，来方便之后的数据处理。为了得到索引，我们将数据集里所有不同字符取出来，
然后将其逐一映射到索引来构造词典。接着，打印vocab_size，即词典中不同字符的个数，又称词典大小。
'''
idx_to_char = list(set(corpus_chars))
print(set(corpus_chars))   # set() 函数创建一个无序不重复元素集合
print(list(set(corpus_chars)))  # 将集合转换为列表
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])   # dict() 函数用于创建一个字典。以可迭代的方式
vocab_size = len(char_to_idx)
print(vocab_size)  # 1027
print(char_to_idx)
print(char_to_idx[' '])
print('-'*100)

'''
之后，将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引。
'''
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)

'''
时序数据的采样:
在训练中我们需要每次随机读取小批量样本和标签。与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。
假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”。
我们有两种方式对时序数据进行采样，分别是随机采样和相邻采样。
'''
'''
(1) 随机采样
下面的代码每次从数据里随机采样一个小批量。其中批量大小batch_size指每个小批量的样本数，num_steps为每个样本所包含的时间步数。 
在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量
最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):  #  变量由from module import *不能导入进来
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        # 例：即“想”“要”“有”“直”“升”。
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        # 该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”。
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

'''
让我们输入一个从0到29的连续整数的人工序列。设批量大小和时间步数分别为2和6。打印随机采样每次读取的小批量样本的输入X和标签Y。
可见，相邻的两个随机小批量在原始序列上的位置不一定相毗邻。
'''
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')

print('-'*100)
'''
(2) 相邻采样:
除对原始序列做随机采样之外，我们还可以令相邻的两个随机小批量在原始序列上的位置相毗邻。这时候，我们就可以用一个小批量最终时间步的隐藏状态
来初始化下一个小批量的隐藏状态，从而使下一个小批量的输出也取决于当前小批量的输入，并如此循环下去。这对实现循环神经网络造成了两方面影响：
一方面， 在训练模型时，我们只需在每一个迭代周期开始时初始化隐藏状态；另一方面，当多个相邻小批量通过传递隐藏状态串联起来时，模型参数的梯度
计算将依赖所有串联起来的小批量序列。同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。 为了使模型参数的梯度计算只依赖一次迭代读取
的小批量序列，我们可以在每次读取小批量前将隐藏状态从计算图中分离出来。我们将在下一节（循环神经网络的从零开始实现）的实现中了解这种处理方式。
'''
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    # print(data_len)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    print(indices)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')


'''
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:", res)
g = foo()
print(next(g))
print("*" * 20)
print(next(g))
'''
'''
1.程序开始执行以后，因为foo函数中有yield关键字，所以foo函数并不会真的执行，而是先得到一个生成器g(相当于一个对象)

2.直到调用next方法，foo函数正式开始执行，先执行foo函数中的print方法，然后进入while循环

3.程序遇到yield关键字，然后把yield想想成return,return了一个4之后，程序停止，并没有执行赋值给res操作，此时next(g)语句执行完成，所以输出的前两行（第一个是while上面的print的结果,第二个是return出的结果）是执行print(next(g))的结果，

4.程序执行print("*"*20)，输出20个*

5.又开始执行下面的print(next(g)),这个时候和上面那个差不多，不过不同的是，这个时候是从刚才那个next程序停止的地方开始执行的，也就是要执行res的赋值操作，这时候要注意，这个时候赋值操作的右边是没有值的（因为刚才那个是return出去了，并没有给赋值操作的左边传参数），所以这个时候res赋值是None,所以接着下面的输出就是res:None,

6.程序会继续在while里执行，又一次碰到yield,这个时候同样return 出4，然后程序停止，print函数输出的4就是这次return出的4.

yield 和 return 的关系和区别了，带 yield 的函数是一个生成器，而不是一个函数了，这个生成器有一个函数就是 next 函数，
next 就相当于“下一步”生成哪个数，这一次的 next 开始的地方是接着上一次的 next 停止的地方执行的，所以调用 next 的时候，
生成器并不会从 foo 函数的开始执行，只是接着上一步停止的地方开始，然后遇到 yield 后，return 出要生成的数，此步就结束。
'''