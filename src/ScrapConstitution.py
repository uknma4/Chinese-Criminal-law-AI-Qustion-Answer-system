import re
import json
import docx
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch, helpers

# 定义一个函数，从 DOCX 文件中读取所有文本
def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # 过滤掉空行
            full_text.append(para.text.strip())
    return "\n".join(full_text)

# 从 "宪法.docx" 中获取法律文档字符串
legal_document = read_docx("/home/eddie/script/law_ask_answer/llama3+ES_Python/Constitution.docx")

# 加载中文向量模型
model_path = "/home/eddie/Share_File/shibing624-text2vec-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
sentence_model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 定义正则表达式模式
chapter_pattern = r"^(第[一二三四五六七八九十百千万]+章\s+.*)$|^(序\u3000\u3000言)$"#应该不可能到亿，注意在正则表达“|”代表“或”
article_pattern = r"^(第[一二三四五六七八九十百千万]+条)\s+(.*)$"#应该不可能到亿

# 将文档按行拆分
lines = legal_document.splitlines()

# 初始化法律文档的整体结构
legal_json = {
  
    "chapters": []
}

current_chapter = None
n=0
# 遍历每一行，根据正则表达式判断是章节还是条文
for line in lines:
    line = line.strip()
    if not line:
        continue

    # 如果匹配章节标题
    
    chapter_match = re.match(chapter_pattern, line)
    if chapter_match:
        current_chapter = {
            "chapter_title": chapter_match.group(0) if line== "序\u3000\u3000言" else chapter_match.group(1),
            "articles": []
        }
        legal_json["chapters"].append(current_chapter)
        continue
        
        
    if legal_json["chapters"] and legal_json["chapters"][-1]["chapter_title"] == "序\u3000\u3000言" and current_chapter is not None:
          # 向量化条文内容
        n=n+1
        vector = sentence_model.encode(line).tolist()
        article_entry = {
             "article_number": "序言" + str(n), ##_id 不能重复否则上传后会覆盖ES
            "content": line,
            "vector": vector
        }
        current_chapter["articles"].append(article_entry)
        continue

    # 如果匹配条文
    article_match = re.match(article_pattern, line)
    if article_match and current_chapter is not None:
        article_number = article_match.group(1)
        content = article_match.group(2)
        # 向量化条文内容
        vector = sentence_model.encode(content).tolist()
        article_entry = {
            "article_number": article_number,
            "content": content,
            "vector": vector
        }
        current_chapter["articles"].append(article_entry)
        continue
 


# 将法律文档保存为 JSON 文件

# json_file_path = "宪法.json"
# with open(json_file_path, 'w', encoding='utf-8') as f:
#     json.dump(legal_json, f, ensure_ascii=False, indent=2)

# 连接到 Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 创建索引（如果不存在）
index_name = 'constitution_documents'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# 准备批量上传的数据
actions = []

# 定义文档元数据
document_metadata = {
    "document_title": "中华人民共和国宪法",
    "document_subtitle": "（1982年12月4日第五届全国人民代表大会第五次会议通过　1982年12月4日全国人民代表大会公告公布施行\n根据1988年4月12日第七届全国人民代表大会第一次会议通过的《中华人民共和国宪法修正案》、1993年3月29日第八届全国人民代表大会第一次会议通过的《中华人民共和国宪法修正案》、1999年3月15日第九届全国人民代表大会第二次会议通过的《中华人民共和国宪法修正案》、2004年3月14日第十届全国人民代表大会第二次会议通过的《中华人民共和国宪法修正案》和2018年3月11日第十三届全国人民代表大会第一次会议通过的《中华人民共和国宪法修正案》修正）",
    "table_of_contents": [
        {"chapter": "序言", "title": "序言"},
        {"chapter": "第一章", "title": "总纲"},
        {"chapter": "第二章", "title": "公民的基本权利和义务"},
        {"chapter": "第三章", "title": "国家机构", 
         "subchapters": [
            {"chapter": "第一节", "title": "全国人民代表大会"},
            {"chapter": "第二节", "title": "中华人民共和国主席"},
            {"chapter": "第三节", "title": "国务院"},
            {"chapter": "第四节", "title": "中央军事委员会"},
            {"chapter": "第五节", "title": "地方各级人民代表大会和地方各级人民政府"},
            {"chapter": "第六节", "title": "民族自治地方的自治机关"},
            {"chapter": "第七节", "title": "监察委员会"},
            {"chapter": "第八节", "title": "人民法院和人民检察院"}
        ]},
        {"chapter": "第四章", "title": "国旗、国歌、国徽、首都"}
    ]
}

action1 = {
            "_op_type": "index",
            "_index": index_name,
            "_id": "document_metadata",
            "_source": {
            
                **document_metadata,  # 添加文档元数据
            }
        }
actions.append(action1)
# 遍历章节和条文，构建上传数据
for chapter in legal_json["chapters"]:
    for article in chapter["articles"]:
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": f"{article['article_number']}", #_id 不能重复否则上传不了ES
            "_source": {
               
                "chapter_title": chapter["chapter_title"],
                "article_number": article["article_number"],
                "content": article["content"],
                "vector": article["vector"]
            }
        }
        actions.append(action)

# 批量上传数据到 Elasticsearch
helpers.bulk(es, actions)

print("数据已成功上传到 Elasticsearch。")
