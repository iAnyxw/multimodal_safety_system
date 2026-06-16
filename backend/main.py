# backend/main.py
"""
多模态安全审核系统 - 一键启动入口

用法（在项目根目录执行）:
    python -m backend.main
    python -m backend.main --port 8000
"""

import argparse
import uvicorn

from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description="Multimodal Safety System")
    parser.add_argument("--host", default=settings.API_HOST, help="绑定 IP")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="绑定端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    print(f"🚀 Starting server at http://{args.host}:{args.port}")
    print(f"📋 API docs at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
