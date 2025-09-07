import re
import json
import docx
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch, helpers
import logging


def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # 过滤掉空行
            full_text.append(para.text.strip())
    return "\n".join(full_text)


# 从 "刑法.docx" 中获取法律文档字符串
legal_document = read_docx("/home/eddie/script/law_ask_answer/llama3+ES_Python/Criminal_Law.docx")

# 加载中文向量模型
model_path = "/home/eddie/Share_File/shibing624-text2vec-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
sentence_model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 定义正则表达式模式
chapter_pattern = r"^(第[零一二三四五六七八九十百千万]+编)\s+(.*)$|^(附\u3000\u3000则)$"  # 匹配“总则”和“分则”
section_pattern = r"^(第[零一二三四五六七八九十百千万]+章)\s+(.*)$"  # 匹配章
subsection_pattern = r"^(第[零一二三四五六七八九十百千万]+节)\s+(.*)$"  # 匹配节
article_pattern = r"^(第[零一二三四五六七八九十百千万]+条(?:[之零一二三四五六七八九十百千万]?)?)\s+(.*)$"  # article_pattern = r"^(第[零一二三四五六七八九十百千万]+条)\s+(.*)$"  # 匹配条文  [之零一二三四五六七八九十百千万]

# 将文档按行拆分
lines = legal_document.splitlines()

# 初始化法律文档的整体结构
legal_json = {
    "chapter": []
}

current_chapter = None
current_section = None
current_subsection = None
# line_number = 0
# section_line_number = 0
# subsection_line_number = 0
# article_line_number = 0
# max_section_line_number = 0
# max_subsection_line_number=0
# 用于保存当前内容的缓存
content_buffer = []



def save_buffer_to_current():
    """Save buffered content to the most specific level (article, subsection, section, chapter)."""
    global content_buffer

    if content_buffer:
        if current_subsection and current_subsection.get("article"):
            # Save buffered content to the last article in the current subsection
            current_subsection["article"][-1]["article_content"] = "\n".join(content_buffer)
        elif current_section and current_section.get("article"):
            # Save buffered content to the last article in the current section
            current_section["article"][-1]["article_content"] = "\n".join(content_buffer)
        elif current_chapter and current_chapter.get("article"):
            # Save buffered content to the last article in the current chapter
            current_chapter["article"][-1]["article_content"] = "\n".join(content_buffer)
        elif current_subsection:
            # If no articles, save to the subsection
            current_subsection["content"] = "\n".join(content_buffer)
        elif current_section:
            # If no articles or subsections, save to the section
            current_section["content"] = "\n".join(content_buffer)
        elif current_chapter:
            # If no articles, subsections, or sections, save to the chapter
            current_chapter["content"] = "\n".join(content_buffer)

        # Clear the buffer after saving
        content_buffer.clear()


# 更新后的内容解析逻辑
for line in lines:
    line = line.strip()
    if not line:
        continue

    # Match chapter titles
    chapter_match = re.match(chapter_pattern, line)
    if chapter_match:
        save_buffer_to_current()  # Save content before starting a new chapter
        title_content = chapter_match.group(0) if line == "附\u3000\u3000则" else chapter_match.group(1)
        title_content = title_content.replace("\u3000\u3000", "")
        vector = sentence_model.encode(title_content).tolist()
        current_chapter = {
            "chapter_title": chapter_match.group(0) if line == "附\u3000\u3000则" else chapter_match.group(1),
            "chapter_title_vector": vector,
            "section": [],
            "subsection": [],
            "article": [],
            "content": ""  # Save chapter-level content
        }

        legal_json["chapter"].append(current_chapter)
        current_section = None
        current_subsection = None
        continue

    # Match section titles
    section_match = re.match(section_pattern, line)
    if section_match and current_chapter is not None:
        save_buffer_to_current()  # Save content before starting a new section
        title_content = section_match.group(0)
        vector = sentence_model.encode(title_content).tolist()
        current_section = {
            "sections_title": section_match.group(0),
            "section_title_vector": vector,
            "subsection": [],
            "article": [],
            "content": ""  # Save section-level content
        }
        current_chapter["section"].append(current_section)
        current_subsection = None
        continue

    # Match subsection titles
    subsection_match = re.match(subsection_pattern, line)
    if subsection_match and current_section is not None:
        save_buffer_to_current()  # Save content before starting a new subsection
        title_content = subsection_match.group(0)
        vector = sentence_model.encode(title_content).tolist()
        current_subsection = {
            "subsections_title": subsection_match.group(0),
            "subsection_title_vector": vector,
            "article": [],
            "content": ""  # Save subsection-level content
        }
        current_section["subsection"].append(current_subsection)
        continue

    # Match articles
    article_match = re.match(article_pattern, line)
    if article_match:
        save_buffer_to_current()  # Save buffered content to the previous article
        article_number = article_match.group(1)
        title = article_match.group(0)
        vector = sentence_model.encode(title).tolist()

        # Create a new article entry
        article_entry = {
            "article_number": article_number,
            "article_title": title,
            "article_title_vector": vector,
            "article_content": "",  # Content will be populated later
            "article_content_vector": ""
        }
        if current_subsection is not None:
            current_subsection["article"].append(article_entry)
        elif current_section is not None:
            current_section["article"].append(article_entry)
        elif current_chapter is not None:
            current_chapter["article"].append(article_entry)
        continue
# Add lines to the content buffer
    content_buffer.append(line)
# Save the last buffered content to the last article
if content_buffer:
    if current_subsection and current_subsection.get("article"):
        current_subsection["article"][-1]["article_content"] = "\n".join(content_buffer)
    elif current_section and current_section.get("article"):
        current_section["article"][-1]["article_content"] = "\n".join(content_buffer)
    elif current_chapter and current_chapter.get("article"):
        current_chapter["article"][-1]["article_content"] = "\n".join(content_buffer)
    content_buffer.clear()


def vectorize_articles(chapters):
    for chapter in chapters:
        for section in chapter.get("section", []):
            for subsection in section.get("subsection", []):
                for article in subsection.get("article", []):
                    article["article_content_vector"] = sentence_model.encode(article["article_content"]).tolist()
            for article in section.get("article", []):
                article["article_content_vector"] = sentence_model.encode(article["article_content"]).tolist()
        for article in chapter.get("article", []):
            article["article_content_vector"] = sentence_model.encode(article["article_content"]).tolist()


vectorize_articles(legal_json["chapter"])

# Save the final JSON file
json_file_path = "刑法.json"
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(legal_json, f, ensure_ascii=False, indent=2)

# 连接到 Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# 创建索引（如果不存在）
index_name = 'crime_documents'
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# 准备批量上传的数据
actions = []

# 定义文档元数据
document_metadata = {
    "document_title": "中华人民共和国刑法",
    "document_subtitle": "（1979年7月1日第五届全国人民代表大会第二次会议通过　1997年3月14日第八届全国人民代表大会第五次会议修订）",
    "table_of_contents": [
        {"chapter": "第一编", "title": "总则", "subchapters": [
            {"section": "第一章", "title": "刑法的任务、基本原则和适用范围"},
            {"section": "第二章", "title": "犯罪", "subsection": [
                {"subsection": "第一节", "title": "犯罪和刑事责任"},
                {"subsection": "第二节", "title": "犯罪的预备、未遂和中止"},
                {"subsection": "第三节", "title": "共同犯罪"},
                {"subsection": "第四节", "title": "单位犯罪"}
            ]},
            {"section": "第三章", "title": "刑罚", "subsection": [
                {"subsection": "第一节", "title": "刑罚的种类"},
                {"subsection": "第二节", "title": "管制"},
                {"subsection": "第三节", "title": "拘役"},
                {"subsection": "第四节", "title": "有期徒刑、无期徒刑"},
                {"subsection": "第五节", "title": "死刑"},
                {"subsection": "第六节", "title": "罚金"},
                {"subsection": "第七节", "title": "剥夺政治权利"},
                {"subsection": "第八节", "title": "没收财产"}
            ]},
            {"section": "第四章", "title": "刑罚的具体运用", "subsection": [
                {"subsection": "第一节", "title": "量刑"},
                {"subsection": "第二节", "title": "累犯"},
                {"subsection": "第三节", "title": "自首和立功"},
                {"subsection": "第四节", "title": "数罪并罚"},
                {"subsection": "第五节", "title": "缓刑"},
                {"subsection": "第六节", "title": "减刑"},
                {"subsection": "第七节", "title": "假释"},
                {"subsection": "第八节", "title": "时效"}
            ]},
            {"section": "第五章", "title": "其他规定"}
        ]},
        {"chapter": "第二编", "title": "分则", "subchapters": [
            {"section": "第一章", "title": "危害国家安全罪"},
            {"section": "第二章", "title": "危害公共安全罪"},
            {"section": "第三章", "title": "破坏社会主义市场经济秩序罪", "subsection": [
                {"subsection": "第一节", "title": "生产、销售伪劣商品罪"},
                {"subsection": "第二节", "title": "走私罪"},
                {"subsection": "第三节", "title": "妨害对公司、企业的管理秩序罪"},
                {"subsection": "第四节", "title": "破坏金融管理秩序罪"},
                {"subsection": "第五节", "title": "金融诈骗罪"},
                {"subsection": "第六节", "title": "危害税收征管罪"},
                {"subsection": "第七节", "title": "侵犯知识产权罪"},
                {"subsection": "第八节", "title": "扰乱市场秩序罪"}
            ]},
            {"section": "第四章", "title": "侵犯公民人身权利、民主权利罪"},
            {"section": "第五章", "title": "侵犯财产罪"},
            {"section": "第六章", "title": "妨害社会管理秩序罪", "subsection": [
                {"subsection": "第一节", "title": "扰乱公共秩序罪"},
                {"subsection": "第二节", "title": "妨害司法罪"},
                {"subsection": "第三节", "title": "妨害国（边）境管理罪"},
                {"subsection": "第四节", "title": "妨害文物管理罪"},
                {"subsection": "第五节", "title": "危害公共卫生罪"},
                {"subsection": "第六节", "title": "破坏环境资源保护罪"},
                {"subsection": "第七节", "title": "走私、贩卖、运输、制造毒品罪"},
                {"subsection": "第八节", "title": "组织、强迫、引诱、容留、介绍卖淫罪"},
                {"subsection": "第九节", "title": "制作、贩卖、传播淫秽物品罪"}
            ]},
            {"section": "第七章", "title": "危害国防利益罪"},
            {"section": "第八章", "title": "贪污贿赂罪"},
            {"section": "第九章", "title": "渎职罪"},
            {"section": "第十章", "title": "军人违反职责罪"}]},
        {"chapter": "第三编", "title": "附则", "subchapters": []}]}



# 添加 document_metadata
action1 = {
    "_op_type": "index",
    "_index": index_name,
    "_id": "document_metadata",
    "_source": {
        **document_metadata,  # 添加文档元数据
    }
}

actions.append(action1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 遍历每个章节（包括附则部分）
for chapter in legal_json["chapter"]:
    # 处理章节级别的文章（如果有）
    for article in chapter.get("article", []):
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_id": f"{chapter['chapter_title']}_{article['article_number']}",
            "_source": {
                "chapter_title": chapter["chapter_title"],
                "chapter_title_vector": chapter["chapter_title_vector"],

                "sections_title": None,
                "section_title_vector": None,
                "subsections_title": None,
                "subsection_title_vector": None,

                "article_title": article["article_title"],
                "article_content": article["article_content"],
                "article_title_vector": article["article_title_vector"],
                "article_content_vector": article["article_content_vector"],
            }
        }
        actions.append(action)

    for section in chapter.get("section", []):
        # 处理 section 中的文章
        for article in section.get("article", []):
            action = {
                "_op_type": "index",
                "_index": index_name,
                "_id": f"{chapter['chapter_title']}_{section['sections_title']}_{article['article_number']}",
                "_source": {
                    "chapter_title": chapter["chapter_title"],
                    "sections_title": section["sections_title"],
                    "section_title_vector": section["section_title_vector"],
                    "subsections_title": None,

                    "article_title": article["article_title"],
                    "article_content": article["article_content"],
                    "article_title_vector": article["article_title_vector"],
                    "article_content_vector": article["article_content_vector"],
                }
            }
            actions.append(action)

        # 处理 section 中的 subsection 和文章
        for subsection in section.get("subsection", []):
            for article in subsection.get("article", []):
                action = {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": f"{chapter['chapter_title']}_{section['sections_title']}_{subsection['subsections_title']}_{article['article_number']}",
                    "_source": {
                        "chapter_title": chapter["chapter_title"],
                        "sections_title": section["sections_title"],
                        "section_title_vector": section["section_title_vector"],
                        "subsections_title": subsection["subsections_title"],
                        "subsection_title_vector": subsection["subsection_title_vector"],

                        "article_title": article["article_title"],
                        "article_content": article["article_content"],
                        "article_title_vector": article["article_title_vector"],
                        "article_content_vector": article["article_content_vector"],
                    }
                }
                actions.append(action)

# 上传数据到 ES
try:
    success, failed = helpers.bulk(es, actions, raise_on_error=False)
    logging.info(f"成功上传文档数: {success}")
    if failed:
        logging.warning(f"失败文档数: {len(failed)}")
        for fail in failed:
            logging.error(f"失败文档: {fail}")
except Exception as e:
    logging.error(f"批量上传失败: {e}")