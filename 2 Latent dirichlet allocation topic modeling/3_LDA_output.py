# 第三步 根据困惑度计算出来的最佳主题数，构建LDA模型，并将预测的主题概率输出
import os.path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
from pprint import pprint
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# 加载数据集
data_all = pd.read_csv("/Users/heyunfan/Desktop/新建文件夹/分词/lda_test2.csv")

# 去除文本中的表情符号（只保留中英文和数字）
def clear_character(sentence):
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern,'',sentence)
    new_sentence = ''.join(line.split())
    return new_sentence
train_text = [clear_character(data_all) for data_all in data_all['review']]

# 添加词典 可以自行在“jiebaDci.txt”中添加希望切出来的词
jieba.load_userdict("/Users/heyunfan/Desktop/新建文件夹/用户评论方法代码/step_3_LDA/jiebaDic3.txt")
# 分词
train_seg_text = [jieba.lcut(s) for s in train_text]
# 加载停用词 可以向文件中添加本研究无意义的词
stop_words_path = "/Users/heyunfan/Desktop/新建文件夹/分词/stop word5.txt"
def get_stop_words():
    return set([item.strip() for item in open(stop_words_path,'r').readlines()])

stopwords = get_stop_words()

# 去除文本中的停用词
def drop_stopwords(line):
    line_clear = []
    for word in line:
        if word in stopwords:
            continue
        line_clear.append(word)
    return line_clear
train_st_text = [drop_stopwords(s) for s in train_seg_text]

# 过滤词长，只保留中文
def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear
train_fine_text = [is_fine_word(s,min_length=2) for s in train_st_text]

#构建bigram 和 trigram 将一些高频词合并成一个单词
bigram = gensim.models.Phrases(train_fine_text,min_count=3,threshold=30)  # threshold是阈值，阈值越高，短语越少
trigram = gensim.models.Phrases(bigram[train_fine_text],threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigram(texts):
    return [trigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)

#构建词典 语料向量化表示
id2word = corpora.Dictionary(train_fine_text)     #create dictionary
texts = train_fine_text                          #create corpus
corpus = [id2word.doc2bow(text) for text in texts]    #term document frequency
#alpha = [0.15,0.1,0.1,0.09,0.08,0.07,0.06,0.05,0.05,0.04,0.04,0.03]
# 构建LDA模型，将选取好的最佳主题数输入参数中num_topics eta就是beta参数
eta_dict = {id2word.token2id['垃圾']: 0.001, id2word.token2id['广告']: 0.001}  # 其他词保持默认
lda_model= gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=12,   # 降低主题数至10（观察右上角填充效果）
                                           random_state=42,      # 固定随机种子确保可复现性
                                           alpha='asymmetric',
                                           eta=0.05,            
                                           passes=30,  # 增加迭代次数至30
                                           iterations=500,         # 提高单次迭代精度
                                           gamma_threshold=0.001,  # 加速收敛
                                           minimum_probability=0.02 )
print(lda_model.alpha)
print(lda_model.eta)
pprint(lda_model.print_topics(num_topics=12,num_words=60)) # 显示每个主题排名前30的关键词
# 获取各个评论在所有主题上的概率
doc_topic = lda_model.get_document_topics(bow=corpus,minimum_probability=0)
topic_score = pd.DataFrame(doc_topic,columns=["Topic {}".format(i) for i in range(0, 12)]) #记得根据主题数修改range
# 结果导出
result = data_all.join(topic_score)
result.to_csv('data_result7.csv',encoding="utf-8")

#获取每个文档的最佳主题，由此可以回溯到某个主题的代表文档，查看其内容
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    temp_list = []  # 用于临时存储每行的数据
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True) #对每个文档的主题按概率排序
        # Get the Dominant topic, Perc Contribution  for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:   #确定最佳主题
                wp = ldamodel.show_topic(topic_num)
                temp_list.append([int(topic_num), round(prop_topic, 4)])
            else:
                break
    sent_topics_df = pd.DataFrame(temp_list, columns=['Dominant_Topic', 'Perc_Contribution'])
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)
#获得每个doc的最可能主题及概率
#df_topic_sents_keywords.to_csv('df_topic_sents_keywords.csv',encoding="utf-8")

#LDAvis可视化，根据可视化结果判断聚类效果，各主题尽量分散不重叠
d = gensimvis.prepare(
    lda_model, corpus, id2word,
    mds='tsne',            # 使用t-SNE替代MDS
    R=30,                  # 提高随机游走步数
    plot_opts={'xlab': 't-SNE1', 'ylab': 't-SNE2'}
)
pyLDAvis.display(d)

pyLDAvis.save_html(d, 'lda_vis7.html')
