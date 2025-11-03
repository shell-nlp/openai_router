
<!-- markdownlint-disable MD033 -->
<h1 align="center">
🚀 OpenAI Router 
</h1>
 
<p align="center">
<b>轻量级、持久化、零配置的 OpenAI API 统一网关</b><br>
一键聚合 vLLM、SGLang、lmdeploy、Ollama…  
</p>
 
<p align="center">
<a href="#"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"></a>
<a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-v0.115+-teal?style=flat-square"></a>
<a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-v5+-orange?style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/SQLite-内置存储-lightgrey?style=flat-square"></a>
<a href="https://pypi.org/project/openai-router/"><img  src="https://img.shields.io/pypi/v/openai-router?style=flat-square&logo=pypi&label=PyPI"></a> 
</p>
 
---
 
## ✨ Features 
| Feature | Description |
|---------|-------------|
| 🌍 统一入口 | `/chat/completions`、`/embeddings`、`/images/generations`… 全部转发 |
| 🧩 多后端 | vLLM、SGLang、lmdeploy、Ollama… 任意组合 |
| 💾 持久化 | SQLite + SQLModel 零配置存储路由 |
| ⚡ 实时流 | SSE & Chunked Transfer 全双工支持 |
| 🎨 Web UI | Gradio 即用的管理面板 |
| 🔍 兼容 OpenAI | SDK / LangChain / AutoGen / LlamaIndex / CrewAI  …等 **一行代码都不用改** |
 
---
 
## 📦 Quick Start 
### Step-1：安装 

#### PyPI（推荐）

```bash 
uv add openai-router -U
```
或者
```bash 
pip install openai-router -U
```


 
### Step-2：启动 
```bash 
openai-router --host localhost --port 8000 
```
浏览器自动打开  
📍 UI：`http://localhost:8000`  
📍 API：`http://localhost:8000/v1`
 
Step-3：添加后端 
在 Web UI 「添加 / 更新」填入：
- 模型名：`gpt-4`
- 后端 URL：`http://localhost:8080/v1`
 
<img src="static/ui.png" width="800">
 
---
 
## 🔧 API Usage 
### **像官方 OpenAI SDK一样调用**
```python 
from openai import OpenAI 
client = OpenAI(
      base_url="http://localhost:8000/v1",
      api_key="sk-dummy"
)
resp = client.chat.completions.create(
      model="gpt-4",
      messages=[{"role":"user","content":"hello"}],
      stream=True 
)
for chunk in resp:
      print(chunk.choices[0].delta.content or "", end="")
```
 
cURL 
```bash 
curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}'
```
 
---
 
## 🗂️ Endpoints 
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Gradio Admin UI |
| `GET` | `/docs` | OpenAPI Swagger |
| `GET` | `/v1/models` | List available models |
 `POST` | `/v1/responses` | Responses API |
| `POST` | `/v1/chat/completions` | Chat completion |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/images/generations` | DALL·E style |
| `POST` | `/v1/audio/transcriptions` | Whisper |
| … | … | All OpenAI endpoints supported |
 
---
 
## ⚙️ Configuration 
CLI Options 
```bash 
python -m openai_router.main --help 
```
| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `localhost` | Bind address |
| `--port` | `8000` | Bind port |
 

---
 
## 🏗️ Architecture 
<img src="static/arch.png" width="800">