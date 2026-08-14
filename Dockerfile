# 使用 Python 3.12 slim 镜像，和项目 pyproject 的 >=3.12 对齐
FROM python:3.12-slim

# 容器内统一工作目录，后续 COPY/RUN/CMD 都基于这里
WORKDIR /app

# 安装 uv，用项目现有的 uv.lock 做可复现依赖安装
RUN pip install --no-cache-dir uv

# 先复制依赖描述文件，利用 Docker 缓存减少重复安装依赖
COPY pyproject.toml uv.lock README.md ./

# 安装项目依赖，不安装 dev 依赖
RUN uv sync --frozen --no-dev --no-install-project

# 复制应用源码
COPY src ./src
COPY scripts ./scripts

# 源码复制完成后，再安装当前项目
RUN uv sync --frozen --no-dev

# FastAPI 服务端口
EXPOSE 8000

# 启动后端服务，容器内必须监听 0.0.0.0
CMD ["uv", "run", "uvicorn", "manufacturing_ai_copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]