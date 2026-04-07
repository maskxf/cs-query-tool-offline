import json
import os
import re
import sys
import math
import webbrowser
import threading
from collections import Counter
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

FAQ_PATH = os.path.join(get_base_path(), "faq.json")

# ========== 中文分词 ==========

def tokenize(text):
    text = text.lower().strip()
    en_words = re.findall(r'[a-z0-9]+', text)
    clean = re.sub(r'[^\u4e00-\u9fff a-z0-9]', ' ', text)
    chars = [c for c in clean if c.strip()]
    tokens = list(chars)
    for i in range(len(chars) - 1):
        if chars[i] != ' ' and chars[i+1] != ' ':
            tokens.append(chars[i] + chars[i+1])
    for i in range(len(chars) - 2):
        if chars[i] != ' ' and chars[i+1] != ' ' and chars[i+2] != ' ':
            tokens.append(chars[i] + chars[i+1] + chars[i+2])
    tokens.extend(en_words)
    return tokens

# ========== BM25 ==========

class BM25Engine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.faqs = []
        self.doc_tokens = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        self.n_docs = 0

    def load(self):
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.doc_tokens = []
        self.doc_freqs = []
        for faq in self.faqs:
            text = " ".join(faq["questions"]) + " " + faq["answer"]
            if faq.get("tags"):
                text += " " + " ".join(faq["tags"])
            tokens = tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_freqs.append(Counter(tokens))
        self.n_docs = len(self.faqs)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.n_docs if self.n_docs else 1
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
        max_score = scores[0][0] if scores and scores[0][0] > 0 else 1
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                break
            norm_score = score / max_score
            faq = self.faqs[idx]
            if norm_score > 0.7:
                tag = "high"
            elif norm_score > 0.4:
                tag = "mid"
            else:
                tag = "low"
            best_q = faq["questions"][0]
            best_q_score = 0
            for q in faq["questions"]:
                q_tok = tokenize(q)
                overlap = len(set(q_tokens) & set(q_tok))
                if overlap > best_q_score:
                    best_q_score = overlap
                    best_q = q
            results.append({
                "score": round(norm_score, 2),
                "tag": tag,
                "category": faq["category"],
                "matched_q": best_q,
                "answer": faq["answer"],
            })
        return results

engine = BM25Engine()

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>客服查询工具</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
.container { max-width: 800px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin: 20px 0; font-size: 24px; }
.search-box { display: flex; gap: 10px; margin-bottom: 20px; }
#query { flex: 1; padding: 12px 16px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px; outline: none; }
#query:focus { border-color: #4a90d9; }
#btn { padding: 12px 24px; font-size: 16px; background: #4a90d9; color: white; border: none; border-radius: 8px; cursor: pointer; }
#btn:hover { background: #357abd; }
.status { text-align: center; color: #666; margin-bottom: 15px; font-size: 14px; }
.result { background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.tag-high { color: #22c55e; font-weight: bold; }
.tag-mid { color: #f59e0b; font-weight: bold; }
.tag-low { color: #ef4444; font-weight: bold; }
.category { background: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 4px; font-size: 13px; }
.matched-q { color: #666; font-size: 14px; margin-bottom: 8px; }
.answer { background: #f8f9fa; padding: 12px; border-radius: 6px; line-height: 1.6; border-left: 3px solid #4a90d9; }
.copy-btn { margin-top: 8px; padding: 4px 12px; font-size: 13px; background: #f0f0f0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
.copy-btn:hover { background: #e0e0e0; }
#results { min-height: 100px; }
.empty { text-align: center; color: #999; padding: 40px; }
</style>
</head>
<body>
<div class="container">
<h1>🎧 客服查询工具</h1>
<div class="status" id="status">加载中...</div>
<div class="search-box">
<input type="text" id="query" placeholder="输入客户问题，例如：头盔怎么选尺码？" autofocus>
<button id="btn" onclick="doSearch()">🔍 查询</button>
</div>
<div id="results"><div class="empty">输入问题后点击查询</div></div>
</div>
<script>
function init() {
    fetch('/api/status').then(r=>r.json()).then(d=>{
        document.getElementById('status').textContent = d.message;
    });
}
function doSearch() {
    var q = document.getElementById('query').value.trim();
    if (!q) return;
    document.getElementById('results').innerHTML = '<div class="empty">查询中...</div>';
    fetch('/api/search?q=' + encodeURIComponent(q)).then(r=>r.json()).then(d=>{
        if (!d.results || d.results.length === 0) {
            document.getElementById('results').innerHTML = '<div class="empty">未找到匹配结果</div>';
            return;
        }
        var html = '';
        var tagMap = {high: '高度匹配 ✅', mid: '参考回复 ⚠️', low: '匹配度低 ❌'};
        var tagClass = {high: 'tag-high', mid: 'tag-mid', low: 'tag-low'};
        for (var i = 0; i < d.results.length; i++) {
            var r = d.results[i];
            html += '<div class="result">';
            html += '<div class="result-header"><span class="' + tagClass[r.tag] + '">' + tagMap[r.tag] + ' (' + r.score + ')</span>';
            html += '<span class="category">' + r.category + '</span></div>';
            html += '<div class="matched-q">匹配问法：' + r.matched_q + '</div>';
            html += '<div class="answer">' + r.answer + '</div>';
            html += '<button class="copy-btn" onclick="copyText(this)">📋 复制回复</button>';
            html += '</div>';
        }
        document.getElementById('results').innerHTML = html;
    });
}
function copyText(btn) {
    var text = btn.previousElementSibling.textContent;
    navigator.clipboard.writeText(text).then(function(){
        btn.textContent = '✅ 已复制';
        setTimeout(function(){ btn.textContent = '📋 复制回复'; }, 1500);
    });
}
document.getElementById('query').addEventListener('keydown', function(e){
    if (e.key === 'Enter') doSearch();
});
init();
</script>
</body>
</html>'''

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            msg = json.dumps({"message": f"✅ 知识库已加载：{engine.n_docs} 条FAQ（离线模式）"}, ensure_ascii=False)
            self.wfile.write(msg.encode('utf-8'))
        elif self.path.startswith('/api/search'):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            q = params.get('q', [''])[0]
            results = engine.search(q) if q else []
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"results": results}, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # 静默日志

def main():
    try:
        n = engine.load()
        print(f"✅ 知识库已加载：{n} 条FAQ")
    except FileNotFoundError:
        print("❌ 未找到 faq.json，请确认文件与程序在同一目录")
        input("按回车退出...")
        return
    except Exception as e:
        print(f"❌ 加载失败：{e}")
        input("按回车退出...")
        return

    port = 7861
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f"🌐 已启动，浏览器访问 http://127.0.0.1:{port}")
    print("关闭此窗口即可停止服务")
    threading.Timer(1, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
