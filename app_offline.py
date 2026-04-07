import json
import os
import re
import sys
import math
from collections import Counter
import gradio as gr

# ========== 路径处理（兼容 PyInstaller 打包） ==========

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

FAQ_PATH = os.path.join(get_base_path(), "faq.json")

# ========== 中文分词（简易版，无外部依赖） ==========

def tokenize(text):
    """简易中文分词：按标点切句，再做 bigram + 单字 + 英文词"""
    text = text.lower().strip()
    # 提取英文/数字词
    en_words = re.findall(r'[a-z0-9]+', text)
    # 去掉标点，保留中文和空格
    clean = re.sub(r'[^\u4e00-\u9fff a-z0-9]', ' ', text)
    chars = [c for c in clean if c.strip()]
    # 单字
    tokens = list(chars)
    # bigram
    for i in range(len(chars) - 1):
        if chars[i] != ' ' and chars[i+1] != ' ':
            tokens.append(chars[i] + chars[i+1])
    # trigram（关键短语）
    for i in range(len(chars) - 2):
        if chars[i] != ' ' and chars[i+1] != ' ' and chars[i+2] != ' ':
            tokens.append(chars[i] + chars[i+1] + chars[i+2])
    tokens.extend(en_words)
    return tokens

# ========== BM25 引擎 ==========

class BM25Engine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.faqs = []
        self.doc_tokens = []    # 每个文档的 token 列表
        self.doc_freqs = []     # 每个文档的词频 Counter
        self.idf = {}
        self.avgdl = 0
        self.n_docs = 0

    def load(self):
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.doc_tokens = []
        self.doc_freqs = []

        for faq in self.faqs:
            # 把所有问法 + 回答 + 标签合并为文档
            text = " ".join(faq["questions"]) + " " + faq["answer"]
            if faq.get("tags"):
                text += " " + " ".join(faq["tags"])
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_freqs.append(Counter(tokens))

        self.n_docs = len(self.faqs)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.n_docs if self.n_docs else 1

        # 计算 IDF
        df = Counter()
        for freq in self.doc_freqs:
            for token in freq:
                df[token] += 1

        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)

        return self.n_docs

    def search(self, query, top_k=3):
        if not self.faqs:
            return []

        q_tokens = tokenize(query)
        scores = []

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            dl = len(self.doc_tokens[i])
            for token in q_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                idf = self.idf.get(token, 0)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tf_norm
            scores.append((score, i))

        scores.sort(reverse=True)

        # 归一化分数到 0-1
        max_score = scores[0][0] if scores and scores[0][0] > 0 else 1

        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                break
            norm_score = score / max_score
            faq = self.faqs[idx]

            if norm_score > 0.7:
                tag = "高度匹配 ✅"
            elif norm_score > 0.4:
                tag = "参考回复 ⚠️"
            else:
                tag = "匹配度低 ❌"

            # 找最匹配的问法
            best_q = faq["questions"][0]
            best_q_score = 0
            for q in faq["questions"]:
                q_tok = tokenize(q)
                overlap = len(set(q_tokens) & set(q_tok))
                if overlap > best_q_score:
                    best_q_score = overlap
                    best_q = q

            results.append({
                "score": norm_score,
                "tag": tag,
                "category": faq["category"],
                "matched_q": best_q,
                "answer": faq["answer"],
            })

        return results

# ========== 初始化 ==========

engine = BM25Engine()

def init_engine():
    try:
        n = engine.load()
        return f"✅ 知识库已加载：{n} 条FAQ（离线模式）"
    except FileNotFoundError:
        return "❌ 未找到 faq.json，请确认文件与程序在同一目录"
    except Exception as e:
        return f"❌ 加载失败：{e}"

# ========== 查询 ==========

def query_faq(question):
    if not question.strip():
        return "请输入客户问题"
    if not engine.faqs:
        return "⚠️ 知识库未加载，请先点击「重载知识库」"

    results = engine.search(question.strip())
    if not results:
        return "未找到匹配结果"

    output = ""
    for i, r in enumerate(results, 1):
        output += f"### 推荐 {i}　{r['tag']}\n"
        output += f"**相似度：**`{r['score']:.2f}`　**分类：**`{r['category']}`\n"
        output += f"**匹配问法：**{r['matched_q']}\n\n"
        output += f"> {r['answer']}\n\n---\n"

    return output.strip("---\n")

def reload_kb():
    return init_engine()

# ========== 界面 ==========

with gr.Blocks(title="客服查询工具") as app:
    gr.Markdown("# 🎧 客服查询工具（离线版）\n输入客户问题，自动匹配知识库推荐回复")

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="客户问题", placeholder="例如：头盔怎么选尺码？", lines=2)
            btn = gr.Button("🔍 查询", variant="primary", size="lg")
            out = gr.Markdown(label="推荐回复")

        with gr.Column(scale=1):
            status = gr.Textbox(label="知识库状态", interactive=False)
            reload_btn = gr.Button("🔄 重载知识库")

    btn.click(fn=query_faq, inputs=inp, outputs=out)
    inp.submit(fn=query_faq, inputs=inp, outputs=out)
    reload_btn.click(fn=reload_kb, outputs=status)
    app.load(fn=init_engine, outputs=status)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
