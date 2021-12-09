# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:41:28 2019

@author: Chinmaya
"""

#%%
import re
import time
import pickle
import os
from statistics import mean

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

from gensim import models, corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
from gensim.matutils import cossim
from gensim.models import Word2Vec

import math

import pandas
import pandas as pd
import numpy
import numpy as np

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,SparsePCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_recall_fscore_support
from sklearn.preprocessing import Normalizer

from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras import models as tf_models

import matplotlib.pyplot as plt
import seaborn as sn

from scipy import sparse

from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
import itertools
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from xgboost import XGBClassifier
#%%

##
def ptime():
    return time.asctime(time.localtime(time.time()))

def remove_stopwords(sentence):
    stop = list(stopwords.words('english'))
    sent_split = list(y for y in sentence.split() if y not in stop)
    return ' '.join(sent_split)

def remove_special_chars(text):
    final = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]
    return " ".join(final)
    
def replace_special_chars(sent):
    chars = [',','.',':',';','-','/']
    for char in chars:
        sent = sent.replace(char,' ')
        
    return sent    

def stripHTMLTags (html):
    text = html
    rules = [
    { r'>\s+' : u'>'},                  # remove spaces after a tag opens or closes
    { r'\s+' : u' '},                   # replace consecutive spaces
    { r'\s*<br\s*/?>\s*' : u'\n'},      # newline after a <br>
    { r'</(div)\s*>\s*' : u'\n'},       # newline after </p> and </div> and <h1/>...
    { r'</(p|h\d)\s*>\s*' : u'\n\n'},   # newline after </p> and </div> and <h1/>...
    { r'<head>.*<\s*(/head|body)[^>]*>' : u'' },     # remove <head> to </head>
    { r'<a\s+href="([^"]+)"[^>]*>.*</a>' : r'\1' },  # show links instead of texts
    { r'[ \t]*<[^<]*?/?>' : u' ' },            # remove remaining tags
    { r'^\s+' : u'' }                   # remove spaces at the beginning
    ]
 
    for rule in rules:
        for (k,v) in rule.items():
            regex = re.compile (k)
            text  = regex.sub (v, text)
 
    special = {
    '&nbsp;' : ' ', '&amp;' : '&', '&quot;' : '"',
    '&lt;'   : '<', '&gt;'  : '>'
    }
 
    for (k,v) in special.items():
        text = text.replace (k, v)
        
    return text


def sent_to_words(sentences):
    for sentence in sentences:
        yield (simple_preprocess(str(sentence), deacc=True))

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_sent(sent):
    lemmatizer = WordNetLemmatizer()
    new_sent = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)]
    return " ".join(new_sent)

def lemmatize_sent_split(sent):
    lemmatizer = WordNetLemmatizer()
    sent = " ".join(sent)
    new_sent = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)]
    return new_sent

def lemmatize_doc(docs):
    new_docs = []
    for doc in docs:
        new_docs.append(lemmatize_sent_split(doc))
    return new_docs

def make_dict(pairs):
    dictionary = {}
    i = 0
    while i < len(pairs):
        if pairs[i] is not None and pairs[i+1] is not None:
            dictionary.update({pairs[i]: pairs[i+1]})

        i = i + 2
    return dictionary

def generate_bigrams(tokens, n=2):

    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ["_".join(ngram) for ngram in ngrams]

def mix_bi_uni(data):
    new_data = []
    flag = 0
    if type(data[0]) != list:
        data = [data]
        flag = 1
    for doc in data:
        bigrams = generate_bigrams(doc)
        doc = doc + bigrams
        new_data.append(doc)
    if flag:
        return new_data[0]
    return new_data    

def clean_sentence(sent):
    s1 = stripHTMLTags(sent.lower())
    s2 = lemmatize_sent(replace_special_chars(s1))
    s3 = remove_stopwords(s2)
    return s3

def clean_sentence_preprocessing(sent):
    s1 = stripHTMLTags(sent.lower())
    s2 = lemmatize_sent(replace_special_chars(s1))
    s3 = remove_stopwords(s2)
    return sent,s3

def find_num_passes(data_length):
    if data_length>=100000:
        return 2

    if data_length<12500:
        return 10
    
    return math.ceil(-0.00007714286*data_length + 9.5)

def make_bigrammer(data_list,min_freq=3):
    data_list_split = list(x.split() for x in data_list)
    bigram = models.Phrases(data_list_split, min_count=min_freq)    
    bigram_mod = Phraser(bigram)    
    return bigram_mod

def prepare_train_data(ans_list,bigram_mod):
    
    data_lemmatized_unigram = list(x.split() for x in ans_list)
    data_words_bigrams = [bigram_mod[doc] for doc in data_lemmatized_unigram]
    
    phrased_data = list(' '.join(doc) for doc in data_words_bigrams)
    
    dictionary = corpora.Dictionary(data_words_bigrams)
    
    dictionary_unigrams = corpora.Dictionary(data_lemmatized_unigram)
    dictionary.merge_with(dictionary_unigrams)
    
    final_data = mix_bi_uni(data_lemmatized_unigram)
    corpus = [dictionary.doc2bow(text) for text in final_data]   
    
    return corpus,dictionary,phrased_data

def bigram_model_index_dict(corpus,dictionary,topics=250):
    
    n = find_num_passes(len(corpus))
    print('\n The number of passes is :- ',n,'\n')
    
    lda = models.LdaMulticore(corpus,num_topics=topics, id2word = dictionary,workers=mp.cpu_count()-1, passes=n)

    print(ptime(), " return lda, dictionary, corpus_lda_tfidf")
    return lda        

def get_unigram_tfidf(ans_list):
    
    Corp = list(x.lower().split() for x in ans_list)
    
    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]
    
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]
    
    return tfidf,dictionary,tfidf_corpus    

def get_bow_corpus(ans_list):
    Corp = list(x.lower().split() for x in ans_list)
    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]
    return dictionary,corpus

def combine_key_value(key_value_dict_list):
    lst = []
    for key in key_value_dict_list.keys():
        
        nkey = remove_stopwords(remove_special_chars(str(key)))
        nvalue = remove_stopwords(remove_special_chars(str(key_value_dict_list[key])))
        
        if nkey.count(' ') > 0:
            nkey = '_'.join(nkey.split())
            
        if nvalue.count(' ') > 0:
            nvalue = '_'.join(nvalue.split())  
            
        lst.append(nkey+'_'+nvalue)
    return lst


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False
    
def make_metrics_dic(metrics_dic):
    new_dic = {}
    for key in metrics_dic.keys():
        if not key.endswith('_unit') and metrics_dic[key] is not None and metrics_dic[str(key)+'_unit'] is not None:
            if isfloat(metrics_dic[key]):
                new_key = key
                new_value = [float(metrics_dic[key]),str(metrics_dic[str(key)+'_unit'])]  
                new_dic.update({new_key:new_value})
            else:
                new_key = key
                new_value = [str(metrics_dic[key]),str(metrics_dic[str(key)+'_unit'])]  
                new_dic.update({new_key:new_value})
    return new_dic

def make_special_metrics_dic(metrics_dic):
    new_dic = {}
    for key in metrics_dic.keys():
        if not key.endswith('_unit') and metrics_dic[key] is not None:
            new_key = key
            if isfloat(metrics_dic[key]):
                new_value = [float(metrics_dic[key]),str(metrics_dic[str(key)+'_unit'])]                  
            else:
                new_value = [str(metrics_dic[key]),str(metrics_dic[str(key)+'_unit'])]  
            new_dic.update({new_key:new_value})
    return new_dic

def size_in_mb(obj):
    from sys import getsizeof as size
    return "{} MB".format(size(obj)/1024)

def mix_bi_uni_query(doc):
    bigrams = generate_bigrams(doc)
    doc = doc + bigrams
    return doc    

def make_desc_query(query):
    query = clean_sentence(query)
    return " ".join(mix_bi_uni_query(query.split()))

def get_top_topics(query,model,dictionary,top = 10):
    q = make_desc_query(query)
    bow = dictionary.doc2bow(q.split())
    topicids = model.get_document_topics(bow)
    topicids = sorted(topicids, key = lambda x: x[1],reverse=True)  
    for z in topicids:
        print([term[0] for term in model.show_topic(z[0], topn=top)],'\n')
    return topicids

def get_common_topics(topicsx,topicsy,model,dictionary,top = 10):
    topicsx = list(j[0] for j in topicsx) 
    topicsy = list(j[0] for j in topicsy) 
    common = list(tid for tid in topicsx if tid in topicsy)
    for z in list(common):
        print([term[0] for term in model.show_topic(z, topn=top)],'\n')
    print('Number of Common Topics are  = ',len(common),'\n\n')
    return common

    
def preprocess_textual_data2(data,types_data):
    title_list = []
    desc_list = []
    product_list =[]
    av_dict_list = []
    type_list = []
    subtype_list = []
    brand_list = []  
    desc_bigrammer_data = []    

    for i,row in enumerate(data):

        desc_full_list = list(clean_sentence(y) for y in row[12:] if y is not None)
        desc = ' '.join(desc_full_list)
        if row[1] is not None and row[1] != '':
            product_list.append(row[0])
            title = row[1].lower()
            title_list.append(title)
            desc_list.append(desc)
            desc_bigrammer_data = desc_bigrammer_data + desc_full_list

            attribute_key_values_list = list(x.lower() if x is not None else x for x in row[2:12])
            
            attribute_key_values_dict = make_dict(attribute_key_values_list)
            
            
            av_dict_list.append(attribute_key_values_dict)
            
            type_list.append(types_data[i][0])
            subtype_list.append(types_data[i][1])
            brand_list.append(types_data[i][2])
            
            if i%1000 == 0:
                print('At the {} record out of {}'.format(i,len(data)))
            
    return title_list,desc_list,product_list,av_dict_list,type_list,subtype_list,brand_list,desc_bigrammer_data

def preprocess_textual_data_dic(data_row,dic):
    row = data_row[0]
    types_row = data_row[1]
    desc_full_list = list(dic[y] for y in row[12:] if y is not None)
    desc = ' '.join(desc_full_list)
    if row[1] is not None and row[1] != '':
        title = row[1].lower()
        attribute_key_values_list = list(x.lower() if x is not None else x for x in row[2:12])
        attribute_key_values_dict = make_dict(attribute_key_values_list)
        return title,desc,row[0],attribute_key_values_dict,types_row[0],types_row[1],types_row[2],desc_full_list   
 
def create_metric_av_lists(dic, fulltype_id_dic, fulltype_key_code_dict, selected_attrs, metric_attrs):
    
    id_prod = dic['id_product']
    pdt_fulltype = fulltype_id_dic[id_prod]
    pdt_special_keys = fulltype_key_code_dict[pdt_fulltype] if pdt_fulltype in fulltype_key_code_dict.keys() else []

    special_av_pairs = []
    special_metric_attrs = []
    for attr in pdt_special_keys:
        if attr+'_unit' in dic.keys() and dic[attr+'_unit'] is not None:
            special_metric_attrs.append(attr)
        else:
            special_av_pairs.append(attr)
    
    selected_attrs2 = list(x for x in selected_attrs if x not in special_av_pairs)
    special_metric_attrs_unit = list(str(x)+'_unit' for x in special_metric_attrs)
    metric_attrs2 = list(x for x in metric_attrs if (x not in special_metric_attrs or x not in special_metric_attrs_unit))
    
    color = None
    color_family = None
    attrs_dic2 = {}
    metric_attrs_dic = {}
    special_attrs_dic = {}
    special_metrics_dic = {}
    
    for key in dic.keys():
        if key == 'colour_name':
            color = '_'.join(sorted(remove_special_chars(dic[key]).lower().split())) if dic[key] is not None else None
        elif key == 'colour_family':
            color_family = '_'.join(sorted(remove_special_chars(dic[key]).lower().split())) if dic[key] is not None else None
        elif key in special_metric_attrs and not key.endswith('_unit'):
            special_metrics_dic.update({key:dic[key]})
            key_unit = str(key)+'_unit'
            value = dic[key_unit] if key_unit in dic.keys() else None
            special_metrics_dic.update({key_unit:value})
        elif key in special_av_pairs:
            if key == 'model_number':
                value = remove_special_chars(dic[key]).lower() if dic[key] is not None else None
                special_attrs_dic.update({key:value})
            else:
                special_attrs_dic.update({key:dic[key]})
        elif key in metric_attrs2:
            metric_attrs_dic.update({key:dic[key]})                        
        elif key in selected_attrs2 and dic[key] is not None:
            attrs_dic2.update({key:dic[key]})

    return color, color_family, make_special_metrics_dic(special_metrics_dic), special_attrs_dic, make_metrics_dic(metric_attrs_dic), attrs_dic2

def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False

def get_in_pixel(qty):
    value = qty[0]
    unit = qty[1]
    units = ['MP','mp','Mp']
    values = [1,1,1]
    return values[units.index(unit)]*value  

def get_in_watt(qty):
    value = qty[0]
    unit = qty[1]
    units = ['watt','kw','kilowatt']
    values = [1,1000,1000]
    return values[units.index(unit)]*value    

def get_in_metre(qty):
    value = qty[0]
    unit = qty[1]
    units = ['km','mile','m','dm','cm','mm','ft','inch','centimeter','centimetre', 'millimeter']
    values = [1000,1609.34,1,0.1,0.01,0.001,0.3048,0.0254,0.01,0.01,0.001]
    return values[units.index(unit)]*value

def get_in_litre(qty):
    value = qty[0]
    unit = qty[1]
    units = ['kl','l','ml','qt','gal','pt','liter','litre']
    values = [1000,1,0.001,0.946353,3.78541,0.568261,1,1]
    return values[units.index(unit)]*value

 
def get_in_gram(qty):
    value = qty[0]
    unit = qty[1]
    units = ['kg','g','mg','cg','oz']
    values = [1000,1,0.001,0.01,28.3495]    
    return values[units.index(unit)]*value

def get_in_megabytes(qty):
    value = qty[0]
    unit = qty[1]
    units = ['MB','GB','TB']
    values = [1,1024,1048576]
    return values[units.index(unit)]*value

def get_in_processorspeed(qty):
    value = qty[0]
    unit = qty[1]
    units = ['Ghz','GHz','GHZ']
    values = [1,1,1]
    return values[units.index(unit)]*value

def get_in_mah(qty):
    value = qty[0]
    unit = qty[1]
    units = ['mAh','mah','MAH','Ah','AH','ah','kAh','kah','KAH']
    values = [1,1,1,1000,1000,1000,1000000,1000000,1000000]
    return values[units.index(unit)]*value

def find_unit_type(unit):
    
    lengths = ['km','mile','m','dm','cm','mm','ft','inch','centimeter','centimetre', 'millimeter']
    weights = ['kg','g','mg','cg','oz']
    litre = ['kl','l','ml','qt','gal','pt','liter','litre']
    power = ['watt','kw','kilowatt']
    memory = ['MB','GB','TB']
    pixels = ['MP','mp','Mp']
    process_speed = ['Ghz','GHz','GHZ']
    battery = ['mAh','mah','MAH','Ah','AH','ah','kAh','kah','KAH']
    
    if unit in lengths:
        return 'length'
    elif unit in weights:
        return 'weight'
    elif unit in litre:
        return 'litre'
    elif unit in memory:
        return 'memory'
    elif unit in power:
        return 'power'
    elif unit in pixels:
        return 'pixel' 
    elif unit in process_speed:
        return 'process_speed' 
    elif unit in battery:
        return 'battery'    
    else:
        return "not_found"

def get_value_as_per_type(lst):
    
    type1 = find_unit_type(lst[1])
    
    if not isfloat(lst[0]):
        return 0
    elif type1 == 'not_found':
        val1 = 0
    elif type1 == 'length':
        val1 = get_in_metre(lst)
    elif type1 == 'litre':
        val1 = get_in_litre(lst)
    elif type1 == 'weight':
        val1 = get_in_gram(lst)
    elif type1 == 'memory':
        val1 = get_in_megabytes(lst)
    elif type1 == 'power':
        val1 = get_in_watt(lst)     
    elif type1 == 'pixel':
        val1 = get_in_pixel(lst)
    elif type1 == 'process_speed':
        val1 = get_in_processorspeed(lst)
    elif type1 == 'battery':
        val1 = get_in_mah(lst)
        
    return val1
  

def get_all_final_attrs():
    
    connection = mysql.connector.connect(host='127.0.0.1',port=3341,database='wecat',user='developer',password='hfN72mk92ngW')
    db_Info = connection.get_server_info()
    print("Connected to MySQL database... MySQL Server version on ",db_Info)
    cursor = connection.cursor()
    
    cmmd = "select code, id_attribute_group from wecat_md.attribute order by id_attribute_group"
    cursor.execute(cmmd)
    att_data = cursor.fetchall()

    cmmd_code = "select id_attribute_group,code from wecat_md.attribute_group"
    cursor.execute(cmmd_code)
    code_id_dic = dict(cursor.fetchall())
    
    attr_group_code_dic = {} 
    for key, group in itertools.groupby(att_data,lambda x:x[1]):
        if key is None:
            key = 'general_none'
        else:
            key = code_id_dic[key]
        if key == 'identifier':
            continue
        attr_group_code_dic[key] = []
        for thing in group:
            attr_group_code_dic[key].append(thing[0])  
    
    all_attrs = list(itertools.chain(*attr_group_code_dic.values()))

    metric_cmmd = "select code from wecat_md.attribute where type ='metric'"
    cursor.execute(metric_cmmd)
    all_metrics = cursor.fetchall()
    
    metrics = [row[0] for row in all_metrics if not row[0].endswith('_unit')]    
    metric_units = [str(x)+'_unit' for x in metrics]
    
    metric_attrs = metrics + metric_units
    selected_attrs = [x for x in all_attrs if x not in metric_attrs]
    
    selected_attrs = list(set(selected_attrs))
    metric_attrs = list(set(metric_attrs))
    
    f_selected = [x for x in selected_attrs if x not in metric_attrs]
    f_metrics = [x for x in metric_attrs if not x.endswith('_unit')]
    
    f_selected.sort()
    f_metrics.sort()

    cmmd = """
    select a.code, ag.code
    from wecat_md.attribute a
    left join wecat_md.attribute_group ag on a.`id_attribute_group`= ag.`id_attribute_group`
    where ag.id_attribute_group is not Null and a.code in ({}) and ag.code <> 'identifier'
    order by ag.id_attribute_group
    """.format(",".join("'" + x + "'" for x in f_selected))
    cursor.execute(cmmd)
    att_data = cursor.fetchall()

    selected_group_code_dic_pre = {} 
    for key, group in itertools.groupby(att_data,lambda x:x[1]):
        selected_group_code_dic_pre[key] = []
        for thing in group:
            selected_group_code_dic_pre[key].append(thing[0])      

    f_selected_group_code_dic = {} 
    general_none = [x for x in attr_group_code_dic['general_none'] if not x.endswith('_unit')]
    f_selected_group_code_dic.update({'general_none':general_none})
    for key in selected_group_code_dic_pre.keys():
        new_key = "_".join(replace_special_chars(key).split())
        f_selected_group_code_dic.update({new_key:selected_group_code_dic_pre[key]})         
    
    return f_selected_group_code_dic,f_metrics
    
def save_av_group_dicts(special_attrs_list,av_dict_list,f_selected_group_code_dic,foldername):
    
    for key_group,attr_list in f_selected_group_code_dic.items():
        dict_corpus = []
        for special_attrs,av_dict in zip(special_attrs_list,av_dict_list):
            new_dict = special_attrs.copy()
            new_dict.update(av_dict)
            for attr in attr_list:
                if attr in special_attrs.keys():
                    dict_corpus.append(special_attrs[attr])
        
        Corp = list(x.split() if x is not None else [] for x in dict_corpus)
        dictionary = corpora.Dictionary(Corp)
        dictionary.save("{}/{}.dictionary".format(foldername,key_group))

def get_product_av_dict_vector(data,f_selected_group_code_dic,f_metrics,foldername):
    
    metrics = data[0]
    special_metrics = data[1]
    special_attrs = data[2]
    av_dict = data[3]
    
    metrics_data = {}
#    product_metric_keys = list(metrics.keys()) + list(special_metrics.keys())
    metric_dict = metrics.copy()
    metric_dict.update(special_metrics)
    for key in f_metrics:
        if key in metric_dict.keys():
#            print("Found the key {}".format(key))
            value = get_value_as_per_type(metric_dict[key])
#            print('Got the value {}'.format(value))
            metrics_data.update({key:value})
#        elif key in special_metrics.keys():
#            metrics_data.update({key:get_value_as_per_type(special_metrics[key])})
        else:
            metrics_data.update({key:0})
    
    new_dict = special_attrs.copy()
    new_dict.update(av_dict)
    
    av_pairs_data = {}
    for key_group,attr_list in f_selected_group_code_dic.items():
#        print("\nIn the group {}".format(key_group))
        attr_values = []
        for attr in attr_list:
            if attr in new_dict.keys():
#                print("Found the attr {}".format(attr))
                try:
                    attr_values.append(new_dict[attr].split() if new_dict[attr] is not None else [])
                except:
#                    print("Got the Exception:- ",new_dict[attr])
                    attr_values.append(str(new_dict[attr]).split() if new_dict[attr] is not None else [])
        
#        print("---Found the values {}".format(attr_values))
        attr_tokens = list(itertools.chain(*attr_values)) 
#        print("---Vector will be formed for tokens {}".format(attr_tokens))
        dictionary = corpora.Dictionary.load("{}/{}.dictionary".format(foldername,key_group))
        bow_vec = dictionary.doc2bow(attr_tokens)
        final_vec = [0]*len(dictionary)
        for tup in bow_vec:
            final_vec[0] = 1
        
        av_pairs_data.update({key_group:final_vec})
        
    return list(metrics_data.values())+list(itertools.chain(*av_pairs_data.values()))
        
def get_all_attrs():

    connection = mysql.connector.connect(host='127.0.0.1',port=3341,database='wecat',user='developer',password='hfN72mk92ngW')
    db_Info = connection.get_server_info()
    print("Connected to MySQL database... MySQL Server version on ",db_Info)
    cursor = connection.cursor()
    
    attrs_cmmd = "select code, id_attribute_group from wecat_md.attribute order by id_attribute_group"
    cursor.execute(attrs_cmmd)
    attrs_data = cursor.fetchall()

    cmmd_code = "select id_attribute_group,code from wecat_md.attribute_group"
    cursor.execute(cmmd_code)
    code_id_dic = dict(cursor.fetchall())

    attr_group_code_dic = {} 
    for key, group in itertools.groupby(attrs_data,lambda x:x[1]):
        if key is None:
            key = 'general_none'
        else:
            key = code_id_dic[key]
        if key == 'identifier':
            continue
        attr_group_code_dic[key] = []
        for thing in group:
            attr_group_code_dic[key].append(thing[0])

    all_attrs = list(itertools.chain(*attr_group_code_dic.values()))

    metric_attrs_cmmd = "select code from wecat_md.attribute where type='metric'"
    cursor.execute(metric_attrs_cmmd)
    metric_attrs_data = cursor.fetchall()
    
    metrics = [row[0] for row in metric_attrs_data if not row[0].endswith('_unit')]    
    metric_units = [str(x)+'_unit' for x in metrics]

    metric_attrs = metrics + metric_units

    pre_considered = ['product_title','product_type','product_subtype','attribute_key_1','attribute_value_1','attribute_key_2','attribute_value_2','attribute_key_3','attribute_value_3','attribute_key_4','attribute_value_4','attribute_key_5','attribute_value_5', 'feature_bullet_1','feature_bullet_2','feature_bullet_3','feature_bullet_4','feature_bullet_5','long_description','id_product_type', 'id_product_subtype','id_brand']
    selected_attrs = [x for x in all_attrs if x not in metric_attrs and x not in pre_considered]

    selected_attrs = list(set(selected_attrs))
    metric_attrs = list(set(metric_attrs))

    return selected_attrs,metric_attrs                
        
    
    
def get_train_data_categorization(title,desc,special_attrs,special_metrics,av_dict):
    row = ""
    row += title
    row += desc
#    row += " ".join([str(j) for j in special_attrs.keys()])
#    row += " ".join([str(j) for j in special_metrics.keys()])
    for dic in [special_attrs,special_metrics,av_dict]:
        for x,y in av_dict.items():
            row  = ' ' + row + ' ' + str(x).lower() + ' ' + str(y).lower()
        
    return row    

    
def get_train_data_categorization2(title,desc):
    row = ""
    row += title + ' '
    row += desc
#    row += " ".join([str(j) for j in special_attrs.keys()])
#    row += " ".join([str(j) for j in special_metrics.keys()])
#    for dic in [special_attrs,special_metrics,av_dict]:
#        for x,y in av_dict.items():
#            row  = ' ' + row + ' ' + str(x).lower() + ' ' + str(y).lower()
    return row.strip(' ')  

def get_train_av_data_categorization2(special_attrs,special_metrics,av_dict):
    special_metrics_transformed = {}
    for key,value in special_metrics.items():
        special_metrics_transformed.update({key:'_'.join(str(x) for x in value)})
    row = ""
    for dic in [special_attrs,special_metrics_transformed,av_dict]:
        for x,y in dic.items():
            row  = ' ' + row + ' ' + str(x).lower() + ' ' + str(y).lower()
    return row.strip(' ')        
            

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        return metrics.accuracy_score(predictions.argmax(axis=-1), valid_y),classifier,predictions
    
    return metrics.accuracy_score(predictions, valid_y),classifier,predictions            

    
def get_lsi_model(ans_list):
    
    Corp = list(x.split() for x in ans_list)
    
    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]

    lsi = models.LsiModel(corpus,num_topics=100)

    return lsi,dictionary,corpus

def sent_vectorizer(sent, model,title_feature_len):
    sent_vec = numpy.zeros(title_feature_len)
    numw = 0
    for w in sent:
        try:
            vc=model[w]
            vc=vc[0:title_feature_len]
            
            sent_vec = numpy.add(sent_vec, vc) 
            numw+=1
        except:
            pass
        
    if numpy.sqrt(sent_vec.dot(sent_vec)) in [0,numpy.nan]:
        return sent_vec
    return sent_vec / numpy.sqrt(sent_vec.dot(sent_vec))

def create_model_architecture(input_size):
    # create input layer 
    print(input_size)
    input_layer = layers.Input(shape =(input_size, ))
    print(input_layer)
    
    # create hidden layer
    hidden_layer1 = layers.Dense(1024, activation="relu",kernel_initializer='random_normal')(input_layer)
    
    hidden_layer2 = layers.Dense(2048, activation="relu",kernel_initializer='random_normal')(hidden_layer1)
    
    hidden_layer3 = layers.Dense(2048, activation="relu",kernel_initializer='random_normal')(hidden_layer2)
    
    hidden_layer4 = layers.Dense(1024, activation="relu",kernel_initializer='random_normal')(hidden_layer3)
    
    hidden_layer5 = layers.Dense(512, activation="relu",kernel_initializer='random_normal')(hidden_layer4)
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid",kernel_initializer='random_normal')(hidden_layer5)

    classifier = tf_models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier
#%%

def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T

#%%

    with open("fulltype_id_dic.pickle",'rb') as f:
        dic_fullltype_ids = pickle.load(f)
        
    
    
    foldername2 = "M:\MBA\BIM_BusinessAnalytics\BIM Project\data"
    
    with open("{}/categorization-data_data.pickle".format(foldername2),'rb') as f:
        data = pickle.load(f)
        
    with open("{}/categorization-data_types_data.pickle".format(foldername2),'rb') as f:
        types_data = pickle.load(f)
        
    with open("{}/categorization-data_title_list.pickle".format(foldername2),'rb') as f:
        title_list = pickle.load(f)
        
    with open("{}/categorization-data_desc_list.pickle".format(foldername2),'rb') as f:
        desc_list = pickle.load(f)
        
    with open("{}/categorization-data_product_list.pickle".format(foldername2),'rb') as f:
        product_list = pickle.load(f)  
    
    with open("{}/categorization-data_fulltype_id_dic.pickle".format(foldername2),'rb') as f:
        fulltype_id_dic = pickle.load(f)  
    
    with open("{}/categorization-data_av_dict_list.pickle".format(foldername2),'rb') as f:
        av_dict_list = pickle.load(f)   
    
    with open("{}/categorization-data_special_metrics_list.pickle".format(foldername2),'rb') as f:
        special_metrics_list = pickle.load(f) 
    
    with open("{}/categorization-data_special_attrs_list.pickle".format(foldername2),'rb') as f:
        special_attrs_list = pickle.load(f)   
    
    with open("{}/categorization-data_metrics_list.pickle".format(foldername2),'rb') as f:
        metrics_list = pickle.load(f)
      
    with open("{}/categorization-data_desc_bigrammer_whole_data.pickle".format(foldername2),'rb') as f:
        desc_bigrammer_whole_data = pickle.load(f)


#%%

        complete_data = pandas.DataFrame()
        complete_data['id_product'] = product_list
        complete_data['title'] = title_list
        complete_data['desc'] = list(desc_list)
        complete_data['special_metrics'] = special_metrics_list 
        complete_data['special_attrs'] = special_attrs_list
        complete_data['av_pairs'] = av_dict_list
        complete_data['desc_bigrammer_whole_data'] = desc_bigrammer_whole_data
        complete_data['metrics'] = metrics_list        
#%%
        fulltypes_with_data = [x for x,y in dic_fullltype_ids.items() if len(y)>=500]

        print(ptime())
        final_training_ids = []
        for fulltype in fulltypes_with_data:
            final_training_ids = final_training_ids + list(random.sample(dic_fullltype_ids[fulltype],500))
            
#            length = 150 if len(dic_fullltype_ids[fulltype]) <= 155 else int(0.75*len(dic_fullltype_ids[fulltype]))
#            final_training_ids = final_training_ids + random.sample(dic_fullltype_ids[fulltype],length)
        print(ptime())


        df = complete_data[complete_data['id_product'].isin(final_training_ids)]


        product_list2 = df['id_product'].tolist()
        title_list2 = df['title'].tolist()
        desc_list2 = df['desc'].tolist()
        special_attrs_list2 = df['special_attrs'].tolist()
        special_metrics_list2 = df['special_metrics'].tolist()
        av_dict_list2 = df['av_pairs'].tolist()
        desc_bigrammer_whole_data2 = df['desc_bigrammer_whole_data'].tolist()
        metrics_list2 = df['metrics'].tolist() 

        desc_bigrammer_data2 = list(itertools.chain(*desc_bigrammer_whole_data2))
#%%
#        print(ptime())
#        categorization_labels = []
#        categorization_labels_and_ids = []
#        for id_product in product_list2:
#            categorization_labels.append(fulltype_id_dic[id_product])
#            categorization_labels_and_ids.append([fulltype_id_dic[id_product],id_product])
#        print(ptime())

#%%
        del product_list,desc_list,title_list,av_dict_list,special_metrics_list,special_attrs_list,desc_bigrammer_whole_data,metrics_list,desc_bigrammer_whole_data2,complete_data,df

#%%
        selected_attrs = ['about_the_author', 'additional_feature_1', 'additional_feature_2', 'additional_feature_3', 'adobe_flash_compatible', 'alcohol_free', 'aperture', 'aperture_f_0_0', 'aromatherapy_type', 'artist_information', 'aspect_ratio', 'assembly', 'assembly_required', 'audio_headphone_type', 'audio_jack', 'author_1', 'author_2', 'author_3', 'author_4', 'author_5', 'author_6', 'autofocus', 'baby_diaper_size', 'bag_style', 'bar_tools', 'base_cap_design', 'base_material', 'base_material_type', 'base_note', 'bath_accessories_type', 'bath_hardware','battery_size_mah' ,'battery_type', 'bedding_size', 'black_out_curtain', 'blu_ray_discs_regions', 'book_description', 'book_format', 'book_format_filter', 'book_subtitle', 'brand_compatibility', 'bra_style_underwear', 'bra_support', 'camera_flash_adjustment', 'camera_lens_adjustment', 'camera_lens_type', 'camera_type', 'carat_weight', 'card_readers', 'card_reader_type', 'care_instructions', 'carrier_furniture_style', 'catalog_tag', 'charging_type', 'cinema_dateslot', 'cinema_seat_type', 'cinema_timeslot', 'cinema_title', 'cinema_venue', 'cleaning_care_accessories', 'closure_fastener', 'closure_fastening', 'closure_type', 'coats_jacket_style_upper_body_wear', 'coffee_tea_type', 'colour_family', 'colour_name', 'comfort_level', 'compatible_with', 'computer_case_type', 'computer_cleaning_type', 'computer_covers_type', 'computer_docking_cradle_type', 'computer_stand_support_type', 'connection_type', 'content_details_1', 'content_details_2', 'content_details_3', 'content_process_nr', 'country_of_origin', 'department', 'dial_colour', 'dial_colour_family', 'dietary_needs', 'director_information', 'dispenser_type', 'display_resolution', 'display_resolution_type', 'display_type', 'dress_style_full_body_wear', 'dubbed', 'edition_number', 'editorial_review', 'editor_1', 'editor_2', 'editor_3', 'energy_used', 'expandable_memory_type', 'exterior_material', 'external_graphics', 'fabric_tie_backs', 'face_material', 'fast_charging', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'filling_material', 'fine_luxury_material', 'fit', 'flash', 'footwear_style', 'formation', 'fragrance_size', 'frame_colour', 'frame_colour_family', 'frame_shape_style', 'genre', 'glass_technology', 'gpc_code', 'gps_enabled', 'graphics_memory_name', 'graphics_memory_version', 'graphic_memory_size', 'grilling_tools', 'gtin', 'handbag_style', 'hands_free_kits_type', 'hat_style', 'hdmi_output', 'heart_middle_note', 'heel_profile', 'heel_type', 'hoseiry_style_underwear', 'inbuilt_flash', 'ink_colour_family', 'installation', 'interchangeable_face_dial', 'interchangeable_lens', 'interchangeable_strap', 'interior_material', 'item_pack_quantity', 'jeans_pants_style_lower_body_wear', 'kitchen_capacity', 'kitchen_storage', 'kitchen_table_linens', 'language', 'laptop_screen_size', 'laptop_type', 'laptop_weight', 'lens_application_type', 'lens_colour', 'lens_colour_family', 'lens_focal_type', 'lens_magnification', 'lens_type', 'lighting_type', 'lined', 'lining_material', 'luggage_accessories_style', 'luggage_style', 'magnifying_factor', 'material', 'material_finish', 'mats_rugs_carpet_design', 'mattress_size', 'max_shutter_speed', 'max_shutter_speed_1_100', 'mega_pixels_mp', 'memory_card_type', 'mfr_min_age', 'mobile_battery_capacity', 'mobile_hotspot_capable', 'mobile_screen_size', 'model_height', 'model_name', 'model_number', 'model_wears', 'monitor_style', 'music_format', 'neckline', 'network', 'network_frequency_band', 'network_type', 'new_carat_weight', 'night_vision_lens', 'number_of_cores', 'number_of_doors', 'number_of_drawers', 'number_of_pages', 'number_of_panels', 'number_of_pieces', 'number_of_seats', 'number_of_shelves', 'number_of_spray_settings', 'number_of_usb_ports', 'occasion', 'operating_system', 'operating_system_name', 'operating_system_number', 'operating_system_version', 'optical_zoom', 'parental_rating', 'partner', 'pattern', 'pen_colour_family', 'pen_colour_name', 'pen_material', 'phone_case_material', 'phone_holder_type', 'pixels_per_inch', 'point_or_nib_type', 'powerbank_capacity', 'primary_camera_feature', 'primary_camera_mp', 'priority', 'priority_type', 'processor_brand', 'processor_name', 'processor_number', 'processor_type', 'processor_version_number_generation', 'production_house', 'product_features_1', 'product_features_2', 'product_features_3', 'product_features_4', 'product_features_5','product_ingredients' ,'publication_date', 'publisher', 'refillable', 'regions', 'retail_price', 'retail_price_ae', 'retail_price_sa', 'safety_feature_type', 'scents_notes', 'screen_features', 'screen_size', 'screen_size_inch', 'sd_card_slot', 'season_code', 'seat_cushion', 'seat_filling', 'secondary_camera_mp', 'secondary_material', 'secondary_reflector', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_type', 'set_includes', 'shape', 'shapewear_style_underwear', 'sheet_type', 'shirt_style_upper_body_wear', 'shorts_style_lower_body_wear', 'sim_count', 'sim_type', 'sizing_standard', 'skirt_dress_length', 'skirt_style_lower_body_wear', 'sleeve_type', 'software_type', 'sole_material', 'specialty', 'special_size_type', 'sport_type', 'standard_dvd_regions', 'stone_gem', 'storage_requirements', 'storage_type', 'strap_material', 'style_of_painting', 'style_or_part_name', 'style_or_part_number', 'subject', 'subject_of_painting', 'subtitles_1', 'subtitles_2', 'subtitles_3', 'sweater_style_upper_body_wear', 'tablet_screen_size', 'target_age_range', 'target_hair_type', 'target_skin_type', 'target_use_application', 'technique', 'toilet_type', 'top_note', 'touch_screen', 'towel_type', 'toys_age_range_new', 'tripod_mount', 'tv_screen_size', 'types_of_bath_care_product', 'types_of_dispenser', 'types_of_ear_nasal_care_product', 'types_of_shower_head', 'types_of_skin_care', 'type_of_accessories', 'type_of_accessory', 'type_of_aquatics', 'type_of_baby_bath', 'type_of_baby_cot_mattress', 'type_of_baby_furnishings', 'type_of_baby_gear_accessories', 'type_of_baby_monitors', 'type_of_baby_safety_equipment', 'type_of_baby_transport', 'type_of_bakeware', 'type_of_bath_pool_toys', 'type_of_batteries_lubricants', 'type_of_bed_bed_frames', 'type_of_bench', 'type_of_bikes_scooters_vehicles_toys', 'type_of_body_parts', 'type_of_brake_parts', 'type_of_camping_furnishings', 'type_of_camping_hiking_accessories', 'type_of_camping_hydration_equipments', 'type_of_camping_kitchen_accessories', 'type_of_candles', 'type_of_candle_holders', 'type_of_car_audio_video', 'type_of_car_care', 'type_of_chair', 'type_of_cleaning_accessories', 'type_of_cleansing_washing_accessories', 'type_of_clocks', 'type_of_coffee_tea_appliance', 'type_of_complexion_cosmetics', 'type_of_console_accessory', 'type_of_console_software', 'type_of_control_aid_accessory', 'type_of_cookware', 'type_of_cosmetic_aids_accessories', 'type_of_curtains_blinds', 'type_of_curtain_lining', 'type_of_cutlery_flatware', 'type_of_dental_care', 'type_of_deodorizer_disinfectant', 'type_of_desk_workstation', 'type_of_diaper', 'type_of_diaper_accessories', 'type_of_dinnerware_serveware', 'type_of_dolls_accessories', 'type_of_drawers', 'type_of_drinkware', 'type_of_educational_developmental_toys', 'type_of_electrical_lighting', 'type_of_engine_interior_cooling', 'type_of_engine_parts', 'type_of_entertainment_unit', 'type_of_exercise_machines_powered_non_powered', 'type_of_exterior_accessories', 'type_of_eye_cosmetics', 'type_of_face_care', 'type_of_false_nails', 'type_of_feeding_training_accessories', 'type_of_feminine_hygiene_product', 'type_of_filing_cabinet', 'type_of_floor_furniture_care', 'type_of_free_weight_dumb_bell_accessories', 'type_of_furniture_cover_or_cloth', 'type_of_gaming_device', 'type_of_gym_accessory', 'type_of_hair_accessory', 'type_of_hair_straightener', 'type_of_hair_styling_product', 'type_of_home_audio_equipment', 'type_of_hygiene_product', 'type_of_indoor_outdoor_garden_toys', 'type_of_insect_pest_control', 'type_of_interior_accessories', 'type_of_kitchen_bathroom_cleaner', 'type_of_large_appliances', 'type_of_laundry_care', 'type_of_lighting', 'type_of_lip_cosmetics', 'type_of_live_plants', 'type_of_mats', 'type_of_mattress', 'type_of_medical_treatment', 'type_of_mirror', 'type_of_model_construction_building_blocks_car_train_sets', 'type_of_musical_toys', 'type_of_nail_aids_accessories', 'type_of_nail_cosmetics_treatments', 'type_of_navigation_accessories', 'type_of_non_electric_appliances', 'type_of_organizers', 'type_of_ornaments', 'type_of_painting', 'type_of_paper_products', 'type_of_personal_fitness_replacement_part_accessory', 'type_of_personal_hygiene_grooming_product', 'type_of_pet', 'type_of_pet_attire', 'type_of_pet_food', 'type_of_pet_grooming_aid', 'type_of_pet_health_hygiene', 'type_of_pet_housing_bedding', 'type_of_picture_frame', 'type_of_portable_audio', 'type_of_puppet_puppet_theatre', 'type_of_puzzles_board_table_card_games', 'type_of_recordable_media', 'type_of_replacement_part', 'type_of_reptile_amphibian', 'type_of_role_play_toys', 'type_of_shelving_unit', 'type_of_shower_care', 'type_of_skin_care', 'type_of_small_appliances', 'type_of_sofa', 'type_of_specialty_electrics', 'type_of_sports_monitor', 'type_of_storage_rail_holder', 'type_of_suspension_parts', 'type_of_system_hardware_portable_non_portable', 'type_of_tables', 'type_of_television', 'type_of_tent_accessories', 'type_of_transmission_parts', 'type_of_travel_accessories', 'type_of_video_receiving_installation', 'type_of_video_recording_playback', 'type_of_wall_art', 'underwear_style_underwear', 'upholstery_type', 'upper_material', 'uv_protection', 'video_feature_1', 'video_feature_2', 'video_format', 'video_game_genre', 'video_genre', 'video_recording_resolution_type', 'video_standard', 'view_finder_type', 'voice_calling_capability', 'waist_type', 'wallet_style', 'wardrobe_design', 'warranty_type', 'warranty_years', 'wash_instructions', 'watch_band_closure', 'watch_band_colour', 'watch_band_colour_family', 'watch_band_material', 'watch_face_colour', 'watch_face_colour_family', 'watch_face_dial_shape', 'watch_face_dial_type', 'watch_movement', 'water_dust_properties', 'width_profile', 'wireless', 'wood_tone', 'wood_type', 'year', 'zoom']
        metric_attrs = ['product_length','product_length_unit','product_width_depth','product_width_depth_unit','product_height','product_height_unit','capacity','capacity_unit','product_weight','product_weight_unit','size','size_unit','battery_size','battery_size_unit','boot_circumference','boot_circumference_unit','boot_shaft_height','boot_shaft_height_unit','dial_face_diameter','dial_face_diameter_unit','heel_height','heel_height_unit','lcd_size','lcd_size_unit','metric','metric_unit','screen_size','screen_size_unit','shipping_height','shipping_height_unit','shipping_length','shipping_length_unit','shipping_weight','shipping_weight_unit','shipping_width_depth','shipping_width_depth_unit','dial_face_diameter','dial_face_diameter_unit','ram_size','ram_size_unit','graphic_memory','graphic_memory_unit','maximum_expandable_memory','maximum_expandable_memory_unit','shaft_height','shaft_height_unit','internal_memory','internal_memory_unit','point_size','point_size_unit','strap_length','strap_length_unit','battery_size', 'battery_size_unit', 'boot_circumference', 'boot_circumference_unit', 'boot_shaft_height', 'boot_shaft_height_unit', 'camera_resolution', 'camera_resolution_unit', 'capacity', 'capacity_unit', 'dial_face_diameter', 'dial_face_diameter_unit', 'diameter', 'diameter_unit', 'graphic_memory', 'graphic_memory_unit', 'laptop_screen_size', 'laptop_screen_size_unit', 'lcd_size', 'lcd_size_unit', 'maximum_expandable_memory', 'maximum_expandable_memory_unit', 'metric', 'metric_unit', 'primary_camera_resolution', 'primary_camera_resolution_unit', 'processor_speed', 'processor_speed_unit', 'product_weight', 'product_weight_unit', 'ram_size', 'ram_size_unit', 'screen_size', 'screen_size_unit', 'secondary_camera_resolution', 'secondary_camera_resolution_unit', 'size', 'size_unit']
           
#%%
#        print(ptime())
#        p = mp.Pool(processes=mp.cpu_count()-1)
#        prod_x = partial(get_product_av_dict_vector, f_selected_group_code_dic = selected_group_attrs_dic, f_metrics = selected_metrics, foldername = train_data_folder)
#        categorization_av_data = p.map(prod_x,zip(metrics_list2,special_metrics_list2,special_attrs_list2,av_dict_list2))
#        p.close()
#        p.join()
#        print(ptime())
#%%
        categorization_data = []
        categorization_labels_and_ids = []
        categorization_av_data = []
        categorization_product_list = []
        
        for id_product,special_attrs,special_metrics,av_dict in zip(product_list2,special_attrs_list2,special_metrics_list2,av_dict_list2):
            categorization_product_list.append(id_product)
            categorization_labels_and_ids.append([fulltype_id_dic[id_product],id_product])
#            categorization_desc_data.append(get_train_data_categorization2(title,desc))
            categorization_av_data.append(get_train_av_data_categorization2(special_attrs,special_metrics,av_dict))
        
        categorization_desc_data = desc_list2
        categorization_title_data = title_list2
#%%

#%%
        
        ## DESC FEATURE EXTRACTION
        
        full_data = list(set(categorization_desc_data))
        
        print(ptime())
        bigrammer_desc = make_bigrammer(desc_bigrammer_data2,2) 
        print(ptime())

        print(ptime())
        (corpus_data,dictionary_data,phrased_data) = prepare_train_data(categorization_desc_data,bigrammer_desc)
        print(ptime())        

        print(ptime())
        (train_set_data,train_dictionary,train_phrased_desc) = prepare_train_data(set(full_data),bigrammer_desc) 
        print(ptime())
        
        print(ptime())
        desc_model = bigram_model_index_dict(train_set_data,train_dictionary) 
        print(ptime())
        
        desc_lda_corpus = desc_model[corpus_data]
        
        lda_desc_features = numpy.zeros([len(desc_lda_corpus),desc_model.num_topics])
        
        print(ptime())
        for i,array in enumerate(desc_lda_corpus):
            for tupl in array:
                lda_desc_features[i,tupl[0]-1] = tupl[1] 
        print(ptime())

        desc_features = sparse.csr_matrix(lda_desc_features)
#%%
        ## TITLE FEATURE EXTRACTION
        
        title_model,title_dic,title_bow_corpus = get_lsi_model(categorization_title_data)
        
        title_lsi_corpus = title_model[title_bow_corpus]
        
        lsi_title_features = numpy.zeros([len(title_lsi_corpus),title_model.num_topics])
       
        ##%%
        print(ptime())
        for i,array in enumerate(title_lsi_corpus):
            for tupl in array:
                lsi_title_features[i,tupl[0]-1] = tupl[1] 
        print(ptime())

        title_features = sparse.csr_matrix(lsi_title_features)
        
#%%
        ###### Count Vectorization
        
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(categorization_av_data)
        av_features =  count_vect.transform(categorization_av_data)
        
#%%
        
        all_train_data = sparse.hstack((title_features,desc_features,av_features))
        
        #%%
        
        # split data into train and test sets
        seed = 7
        test_size = 0.50
        X_train, X_test, y_train_and_ids, y_test_and_ids = train_test_split(all_train_data.tocsc(), categorization_labels_and_ids, test_size=test_size, random_state=seed)        
        
#%%
        
#        ids_train,ids_test,ids_bekar1,ids_bekar2 = train_test_split(categorization_product_list,[1]*len(categorization_product_list),test_size=test_size, random_state=seed)\
        y_train = [x[0] for x in y_train_and_ids]
        train_ids = [x[1] for x in y_train_and_ids]
        y_test = [x[0] for x in y_test_and_ids]
        test_ids = [x[1] for x in y_test_and_ids]

#%%
        start = time.time()
        # fit model no training data
        xgb_model = XGBClassifier(verbosity = 2, nthread = 3, max_depth=4)
        xgb_model.fit(X_train, y_train)
        print(xgb_model)
        end = time.time()
        print(" Total Time taken = {} minutes".format((end-start)/60))

#%%
        y_pred = xgb_model.predict(X_test)
        xgb_predictions = [round(value) for value in y_pred]

        # evaluate predictions
        xgb_accuracy = accuracy_score(xgb_predictions, y_test)
        print("XgBoost Accuracy: %.2f%%" % (xgb_accuracy * 100.0))
        
        ## XgBoost Implementation Ends
#%%
        ## rforest Implemetation
        start = time.time()
        rforest_accuracy,rforest_model,rforest_predictions = train_model(ensemble.RandomForestClassifier(n_estimators=101), X_train, y_train, X_test,y_test)
        print("rforest Accuracy = {}".format(rforest_accuracy))
        end = time.time()
        print(" Total Time taken = {} minutes".format((end-start)/60)) 
#%%
        
        target_names = list(str(x) for x in set(y_test))
#%%

        cm = confusion_matrix(y_test, rforest_predictions)
#%%        
        df_cm = pandas.DataFrame(cm)
        plt.figure(figsize = df_cm.shape)
        sn.heatmap(df_cm, annot=True)

#%%
        report = classification_report(y_test, rforest_predictions, target_names=target_names) 
#%%

        df = pandas.DataFrame(report).transpose()
        
#%%

        df_class_report = pandas_classification_report(y_true=y_test, y_pred=rforest_predictions)
        df_class_report.to_csv("report.csv",sep=',')
        
#%%

        cols = ['precision', 'recall', 'f1-score', 'support']
        for x in cols:
            print(" Mean {} = {}".format(x, mean(df_class_report[x][:189])))
        
        