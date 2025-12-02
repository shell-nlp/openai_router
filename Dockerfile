FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app 

# 设置 UV 镜像源为清华大学源以加速依赖安装
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple


RUN uv init && uv add openai-router==0.1.5

# 把 venv 的 bin 放进 PATH，后面可以直接用 openai-router 
ENV PATH="/app/.venv/bin:$PATH"

CMD ["/bin/bash"]