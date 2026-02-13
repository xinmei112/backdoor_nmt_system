#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMT 后门攻击实验平台 - 启动脚本
带完整错误捕获
"""

import sys
import traceback


def main():
    try:
        print("=" * 60)
        print("🚀 正在启动 NMT 后门攻击实验平台...")
        print("=" * 60)
        print()

        # 检查 Python 版本
        print("📍 Python 版本:", sys.version)
        if sys.version_info < (3, 7):
            print("❌ 错误: 需要 Python 3.7 或更高版本")
            print(f"   当前版本: {sys.version_info.major}.{sys.version_info.minor}")
            return 1
        print("✅ Python 版本检查通过")
        print()

        # 检查依赖包
        print("📦 检查依赖包...")
        required_packages = {
            'flask': 'Flask',
            'flask_cors': 'Flask-CORS',
            'transformers': 'transformers',
            'torch': 'PyTorch',
        }

        missing_packages = []
        for module_name, package_name in required_packages.items():
            try:
                __import__(module_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                print(f"  ❌ {package_name} - 未安装")
                missing_packages.append(package_name)

        if missing_packages:
            print()
            print("❌ 缺少依赖包，请运行以下命令安装:")
            print(f"   pip install {' '.join(missing_packages)}")
            return 1

        print("✅ 所有依赖包已安装")
        print()

        # 导入 Flask app
        print("📥 导入应用模块...")
        try:
            from app import app
            print("✅ 应用模块导入成功")
        except Exception as e:
            print("❌ 导入应用模块失败:")
            print(f"   {type(e).__name__}: {str(e)}")
            print()
            print("详细错误信息:")
            traceback.print_exc()
            return 1

        print()
        print("=" * 60)
        print("🌐 启动 Web 服务器...")
        print("=" * 60)
        print()
        print("📍 访问地址: http://localhost:5000")
        print("📍 按 Ctrl+C 停止服务器")
        print()

        # 启动服务器
        try:
            app.run(
                host='0.0.0.0',
                port=5000,
                debug=True,
                use_reloader=False  # 禁用重载器，避免双重启动
            )
        except KeyboardInterrupt:
            print()
            print("=" * 60)
            print("👋 服务器已停止")
            print("=" * 60)
            return 0
        except Exception as e:
            print()
            print("=" * 60)
            print("❌ 服务器启动失败:")
            print(f"   {type(e).__name__}: {str(e)}")
            print("=" * 60)
            print()
            print("详细错误信息:")
            traceback.print_exc()
            return 1

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 启动过程中发生未知错误:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("=" * 60)
        print()
        print("详细错误信息:")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())