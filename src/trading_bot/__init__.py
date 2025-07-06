"""
Bitcoin Trading Bot System - Made by WilBTC
==========================================

A comprehensive multi-timeframe cryptocurrency trading system with machine learning,
continuous learning, and automated profit maximization.

Version: 3.0 - Production Ready
Author: WilBTC
"""

__version__ = "3.0.0"
__author__ = "WilBTC"
__description__ = "Bitcoin Trading Bot System with AI and Profit Maximization"

# Core imports
from .clients.lnmarkets_client import LNMarketsClient
from .agents.profit_maximizer import ProfitMaximizer
from .api.trading_server import TradingServer

__all__ = [
    "LNMarketsClient",
    "ProfitMaximizer", 
    "TradingServer",
    "__version__",
    "__author__",
    "__description__"
]