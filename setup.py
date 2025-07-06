"""
🔥 BITCOIN TRADING AGENT - SETUP 🔥
Made by WilBTC

The AI that actually makes money.
"""

from setuptools import setup, find_packages

setup(
    name="bitcoin-trading-agent-wilbtc",
    version="3.0.0",
    author="WilBTC",
    author_email="contact@wilbtc.com",
    description="🔥 The AI Trading Agent That Actually Works - Made by WilBTC",
    long_description="""
# 🔥 THE BITCOIN MONEY PRINTER 🔥

The AI trading agent that actually makes money instead of just backtesting fake profits.

## What makes this different?

✅ **ACTUALLY PROFITABLE** in live trading (not just backtests)
✅ **10-SECOND SCANNING** for opportunities  
✅ **AI RISK MANAGEMENT** that protects your stack
✅ **4 SPECIALIZED AGENTS** working 24/7
✅ **SELF-HOSTED** (your keys, your coins, your control)

## Quick Start:

```bash
pip install bitcoin-trading-agent-wilbtc
make install
make run
```

That's it. The AI does the rest.

⚡ Made by WilBTC - Stack Sats, Maximize Profits ⚡
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/WilBtc/TradingAgent",
    project_urls={
        "Bug Reports": "https://github.com/WilBtc/TradingAgent/issues",
        "Documentation": "https://github.com/WilBtc/TradingAgent/wiki",
        "Source": "https://github.com/WilBtc/TradingAgent",
        "Discord": "https://discord.gg/wilbtc",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=1.0.0",
        "redis>=5.0.0",
        "ccxt>=4.0.0",
        "websockets>=12.0",
        "ta-lib>=0.4.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "feedparser>=6.0.0",
        "asyncio-mqtt>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-agent=trading_bot.cli:main",
            "money-printer=trading_bot.agents.profit_maximizer:main",
        ],
    },
    keywords="bitcoin trading ai cryptocurrency bot profit automation lnmarkets",
    license="MIT",
    zip_safe=False,
)