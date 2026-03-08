#!/bin/bash
# 在服务器上执行: bash scripts/start-rag-app.sh
# 使用主机端口 8502，避免与本地 streamlit 冲突

set -e
cd "$(dirname "$0")/.."

echo ">>> 检查并释放 8502 端口..."
PID=$(lsof -t -i:8502 2>/dev/null || true)
if [ -n "$PID" ]; then
  echo "    结束占用进程 PID=$PID"
  kill -9 $PID 2>/dev/null || true
  sleep 1
fi

echo ">>> 删除旧容器（若存在）..."
sudo docker rm -f rag-app 2>/dev/null || true

echo ">>> 确保环境变量文件存在..."
if [ ! -f /opt/rag-app/docker.env ]; then
  echo "    错误: /opt/rag-app/docker.env 不存在，请先创建并填入 DASHSCOPE_API_KEY 和 GEMINI_API_KEY"
  exit 1
fi

echo ">>> 启动容器 (映射 8502:8501)..."
sudo docker run -d \
  --name rag-app \
  -p 8502:8501 \
  --env-file /opt/rag-app/docker.env \
  -v /opt/rag-data/data:/app/data \
  -v /opt/rag-data/chroma_db:/app/chroma_db \
  -v /opt/rag-data/logs:/app/logs \
  rag-app:latest

echo ">>> 检查状态..."
sleep 2
sudo docker ps --filter name=rag-app
echo ""
sudo docker port rag-app
echo ""
echo ">>> 完成。浏览器访问: http://你的ECS公网IP:8502"
echo ">>> 若无法访问，请在阿里云安全组放行 TCP 8502 端口。"
