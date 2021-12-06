from stanfordcorenlp import StanfordCoreNLP
import pickle
from tqdm import tqdm
name = 'math_mcp'

with StanfordCoreNLP('model path and config') as nlp:
    with open('./data/%s.txt' % name, 'r', encoding='utf-8') as f:
        ques_list = f.readlines()[1:]
    result = {}
    for question in tqdm(ques_list):
        ques_id = question.split('\t')[0]
        content = question.split('\t')[1].replace(' ', '').replace('%', '')
        content = content.split('.')
        result[ques_id] = {}
        result[ques_id]['pos_tag'] = [nlp.pos_tag(s) for s in content if s != '']
        result[ques_id]['parse'] = [nlp.parse(s) for s in content if s != '']
        result[ques_id]['dependency_parse'] = [nlp.dependency_parse(s) for s in content if s != '']
pickle.dump(result, open('./data/%s.obj' % name, 'wb'))
