import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor

from deepseek_llm_integration import DeepSeekTradingAI
from redis_cache import TradingCache
from real_technical_indicators import TechnicalIndicators
from lnmarkets_official_client import LNMarketsOfficialClient
from lnmarkets_data_provider import LNMarketsDataProvider


class AgentType(Enum):
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    RISK_MANAGER = "risk_manager"
    NEWS = "news"


@dataclass
class AgentAnalysis:
    agent_type: AgentType
    confidence: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    reasoning: str
    data_sources: List[str]
    timestamp: datetime
    risk_score: float
    metadata: Dict[str, Any]


@dataclass
class TradingDecision:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    risk_assessment: Dict[str, Any]
    agent_consensus: Dict[str, Any]
    timestamp: datetime


class BaseAgent:
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.cache = TradingCache()
        self.logger = logging.getLogger(f"agent_{agent_type.value}")
        self.memory = []  # Store past analyses for learning
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> AgentAnalysis:
        raise NotImplementedError("Subclasses must implement analyze method")
    
    def update_memory(self, analysis: AgentAnalysis, outcome: Optional[Dict[str, Any]] = None):
        """Update agent memory with analysis and outcome for learning"""
        memory_entry = {
            'analysis': analysis,
            'outcome': outcome,
            'timestamp': datetime.now()
        }
        self.memory.append(memory_entry)
        
        # Keep only last 100 entries
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]


class BitcoinFundamentalAgent(BaseAgent):
    """Bitcoin-focused fundamental analysis using free data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.FUNDAMENTAL, config)
        self.data_provider = LNMarketsDataProvider()
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> AgentAnalysis:
        """Analyze Bitcoin fundamentals using free data sources"""
        try:
            # Get Bitcoin fundamental data
            fundamentals = market_data.get('fundamentals', {})
            
            if not fundamentals:
                return self._fallback_analysis(symbol, "No fundamental data available")
            
            # Calculate fundamental score based on available metrics
            fundamental_score = self._calculate_fundamental_score(fundamentals)
            
            # Generate signal based on fundamental analysis
            if fundamental_score > 0.7:
                signal = "BUY"
                confidence = min(fundamental_score, 0.9)
            elif fundamental_score < 0.3:
                signal = "SELL"
                confidence = min(1 - fundamental_score, 0.9)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            reasoning = f"Bitcoin fundamental analysis: Score {fundamental_score:.2f}. "
            
            # Add specific fundamental insights
            if fundamentals.get('market_cap'):
                reasoning += f"Market cap: ${fundamentals['market_cap']/1e9:.1f}B. "
            
            if fundamentals.get('fear_greed_index') is not None:
                fg_index = fundamentals['fear_greed_index']
                reasoning += f"Fear & Greed Index: {fg_index} ({self._interpret_fear_greed(fg_index)}). "
            
            if fundamentals.get('dominance_percentage'):
                reasoning += f"Bitcoin dominance: {fundamentals['dominance_percentage']:.1f}%. "
            
            return AgentAnalysis(
                agent_type=self.agent_type,
                confidence=confidence,
                signal=signal,
                reasoning=reasoning,
                data_sources=["Bitcoin Network", "CoinGecko", "Fear & Greed Index"],
                timestamp=datetime.now(),
                risk_score=1 - confidence,
                metadata={
                    'fundamental_score': fundamental_score,
                    'fear_greed_index': fundamentals.get('fear_greed_index'),
                    'market_cap': fundamentals.get('market_cap'),
                    'circulating_supply': fundamentals.get('circulating_supply')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Bitcoin fundamental analysis failed: {e}")
            return self._fallback_analysis(symbol, "Fundamental analysis error")
    
    def _calculate_fundamental_score(self, fundamentals: Dict[str, Any]) -> float:
        """Calculate fundamental strength score for Bitcoin (0-1)"""
        score = 0.5  # Base score
        
        try:
            # Fear & Greed Index analysis
            fg_index = fundamentals.get('fear_greed_index')
            if fg_index is not None:
                if fg_index < 20:  # Extreme fear - potential buying opportunity
                    score += 0.2
                elif fg_index < 40:  # Fear - good buying opportunity
                    score += 0.1
                elif fg_index > 80:  # Extreme greed - potential selling opportunity
                    score -= 0.2
                elif fg_index > 60:  # Greed - caution
                    score -= 0.1
            
            # Market cap analysis (growth indicates adoption)
            market_cap = fundamentals.get('market_cap')
            if market_cap:
                # Higher market cap generally indicates stability
                if market_cap > 1e12:  # > $1T
                    score += 0.1
                elif market_cap > 5e11:  # > $500B
                    score += 0.05
            
            # Bitcoin dominance (higher dominance can indicate Bitcoin strength)
            dominance = fundamentals.get('dominance_percentage')
            if dominance:
                if dominance > 45:
                    score += 0.1
                elif dominance < 35:
                    score -= 0.05
            
            # Circulating supply vs max supply (scarcity factor)
            circulating = fundamentals.get('circulating_supply')
            max_supply = fundamentals.get('max_supply', 21000000)
            if circulating and max_supply:
                scarcity_ratio = circulating / max_supply
                if scarcity_ratio > 0.95:  # Very scarce
                    score += 0.1
                elif scarcity_ratio > 0.9:  # Becoming scarce
                    score += 0.05
            
        except Exception as e:
            self.logger.warning(f"Error calculating fundamental score: {e}")
        
        return max(0, min(1, score))
    
    def _interpret_fear_greed(self, index: int) -> str:
        """Interpret Fear & Greed Index"""
        if index <= 20:
            return "Extreme Fear"
        elif index <= 40:
            return "Fear"
        elif index <= 60:
            return "Neutral"
        elif index <= 80:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def _fallback_analysis(self, symbol: str, reason: str) -> AgentAnalysis:
        """Fallback analysis when data is unavailable"""
        return AgentAnalysis(
            agent_type=self.agent_type,
            confidence=0.1,
            signal="HOLD",
            reasoning=reason,
            data_sources=[],
            timestamp=datetime.now(),
            risk_score=0.9,
            metadata={}
        )


class CryptoSentimentAgent(BaseAgent):
    """Crypto sentiment analysis using news and free APIs"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.SENTIMENT, config)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.deepseek = DeepSeekTradingAI()
        self.data_provider = LNMarketsDataProvider()
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> AgentAnalysis:
        """Analyze crypto sentiment from news and market indicators"""
        try:
            # Get sentiment data from market data
            sentiment_data = market_data.get('sentiment_analysis', {})
            news_data = market_data.get('news_data', [])
            
            sentiment_scores = []
            data_sources = []
            
            # News sentiment
            news_sentiment = sentiment_data.get('news_sentiment', 0)
            if news_sentiment != 0:
                sentiment_scores.append(news_sentiment)
                data_sources.append("Crypto News")
            
            # Fear & Greed Index
            fg_sentiment = sentiment_data.get('fear_greed_index')
            if fg_sentiment is not None:
                # Convert 0-100 to -1 to 1
                fg_normalized = (fg_sentiment - 50) / 50
                sentiment_scores.append(fg_normalized)
                data_sources.append("Fear & Greed Index")
            
            # AI-powered sentiment analysis
            if news_data:
                ai_sentiment = await self._analyze_ai_sentiment(symbol, news_data[:3])
                if ai_sentiment is not None:
                    sentiment_scores.append(ai_sentiment)
                    data_sources.append("AI Analysis")
            
            # Aggregate sentiment scores
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                confidence = min(0.9, abs(avg_sentiment) + 0.2)
                
                if avg_sentiment > 0.2:
                    signal = "BUY"
                elif avg_sentiment < -0.2:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                    
                reasoning = f"Crypto sentiment analysis: {avg_sentiment:.2f} from {len(sentiment_scores)} sources. "
                reasoning += f"News items analyzed: {len(news_data)}. "
                
                if fg_sentiment is not None:
                    reasoning += f"Fear & Greed: {fg_sentiment} ({self._interpret_fear_greed(fg_sentiment)}). "
                
                return AgentAnalysis(
                    agent_type=self.agent_type,
                    confidence=confidence,
                    signal=signal,
                    reasoning=reasoning,
                    data_sources=data_sources,
                    timestamp=datetime.now(),
                    risk_score=1 - confidence,
                    metadata={
                        'sentiment_score': avg_sentiment,
                        'individual_scores': sentiment_scores,
                        'news_count': len(news_data),
                        'fear_greed_index': fg_sentiment
                    }
                )
            else:
                return self._fallback_analysis(symbol, "No sentiment data available")
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_analysis(symbol, "Sentiment analysis error")
    
    async def _analyze_ai_sentiment(self, symbol: str, news_items: List[Dict]) -> Optional[float]:
        """Use AI to analyze news sentiment"""
        try:
            if not news_items:
                return None
            
            # Prepare news summary for AI
            news_summary = ""
            for item in news_items:
                news_summary += f"- {item.get('title', '')}: {item.get('content', '')[:200]}...\n"
            
            # Use simpler method that doesn't require 'indicators' parameter
            prompt = f"Analyze the sentiment of these Bitcoin news items and return a score from -1 (very negative) to 1 (very positive):\n\n{news_summary}"
            
            # Call DeepSeek with minimal parameters
            try:
                # Use the basic chat completion
                response = await self._call_deepseek_simple(prompt)
                
                # Extract sentiment score from response
                if "positive" in response.lower() or "bullish" in response.lower():
                    return 0.3
                elif "negative" in response.lower() or "bearish" in response.lower():
                    return -0.3
                else:
                    return 0.0
            except Exception as e:
                self.logger.warning(f"DeepSeek AI call failed: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"AI sentiment analysis failed: {e}")
            return None
    
    async def _call_deepseek_simple(self, prompt: str) -> str:
        """Simple DeepSeek API call"""
        try:
            # Use a basic analysis method
            if hasattr(self.deepseek, 'generate_simple_analysis'):
                return await self.deepseek.generate_simple_analysis(prompt)
            else:
                # Fallback to basic text analysis
                return "neutral sentiment detected"
        except Exception as e:
            self.logger.error(f"DeepSeek simple call failed: {e}")
            return "neutral"
    
    def _interpret_fear_greed(self, index: int) -> str:
        """Interpret Fear & Greed Index"""
        if index <= 20:
            return "Extreme Fear"
        elif index <= 40:
            return "Fear"
        elif index <= 60:
            return "Neutral"
        elif index <= 80:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def _fallback_analysis(self, symbol: str, reason: str) -> AgentAnalysis:
        """Fallback analysis when sentiment data is unavailable"""
        return AgentAnalysis(
            agent_type=self.agent_type,
            confidence=0.1,
            signal="HOLD",
            reasoning=reason,
            data_sources=[],
            timestamp=datetime.now(),
            risk_score=0.9,
            metadata={}
        )


class LNMarketsTechnicalAgent(BaseAgent):
    """Technical analysis using LNMarkets data and existing indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.TECHNICAL, config)
        self.technical_indicators = TechnicalIndicators()
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> AgentAnalysis:
        """Enhanced technical analysis using LNMarkets data"""
        try:
            # Get technical analysis data
            technical_data = market_data.get('technical_analysis', {})
            
            if 'error' in technical_data or not technical_data:
                return self._fallback_analysis(symbol, "Insufficient technical data")
            
            # Calculate technical score
            technical_score = self._calculate_technical_score(technical_data)
            
            # Generate signal
            if technical_score > 0.6:
                signal = "BUY"
                confidence = min(technical_score, 0.9)
            elif technical_score < 0.4:
                signal = "SELL"
                confidence = min(1 - technical_score, 0.9)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            reasoning = f"Technical analysis: Score {technical_score:.2f}. "
            
            # Add specific technical insights
            rsi = technical_data.get('rsi')
            if rsi:
                reasoning += f"RSI: {rsi:.1f} ({self._interpret_rsi(rsi)}). "
            
            trend = technical_data.get('trend', 'neutral')
            reasoning += f"Trend: {trend}. "
            
            volatility = technical_data.get('volatility', 0)
            reasoning += f"Volatility: {volatility:.4f}. "
            
            volume_trend = technical_data.get('volume_trend', 'stable')
            reasoning += f"Volume: {volume_trend}."
            
            return AgentAnalysis(
                agent_type=self.agent_type,
                confidence=confidence,
                signal=signal,
                reasoning=reasoning,
                data_sources=["LNMarkets", "Technical Indicators"],
                timestamp=datetime.now(),
                risk_score=1 - confidence,
                metadata={
                    'technical_score': technical_score,
                    'technical_data': technical_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return self._fallback_analysis(symbol, "Technical analysis error")
    
    def _calculate_technical_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate overall technical score (0-1)"""
        score = 0.5  # Base score
        
        try:
            # RSI analysis
            rsi = technical_data.get('rsi')
            if rsi is not None:
                if 30 <= rsi <= 70:
                    score += 0.1  # Neutral RSI is good
                elif rsi < 30:
                    score += 0.2  # Oversold - potential buy
                elif rsi > 70:
                    score -= 0.2  # Overbought - potential sell
            
            # MACD analysis
            macd_line = technical_data.get('macd_line')
            macd_signal = technical_data.get('macd_signal')
            if macd_line is not None and macd_signal is not None:
                if macd_line > macd_signal:
                    score += 0.1  # Bullish MACD
                else:
                    score -= 0.1  # Bearish MACD
            
            # Moving average analysis
            current_price = technical_data.get('current_price')
            sma_20 = technical_data.get('sma_20')
            ema_12 = technical_data.get('ema_12')
            
            if current_price and sma_20:
                if current_price > sma_20:
                    score += 0.1
                else:
                    score -= 0.1
            
            if current_price and ema_12:
                if current_price > ema_12:
                    score += 0.1
                else:
                    score -= 0.1
            
            # Trend analysis
            trend = technical_data.get('trend', 'neutral')
            if trend == 'bullish':
                score += 0.15
            elif trend == 'bearish':
                score -= 0.15
            
            # Volume analysis
            volume_trend = technical_data.get('volume_trend', 'stable')
            if volume_trend == 'increasing':
                score += 0.05  # Increasing volume is generally positive
            elif volume_trend == 'decreasing':
                score -= 0.05
            
            # Support/resistance analysis
            support_resistance = technical_data.get('support_resistance')
            if support_resistance and current_price:
                support, resistance = support_resistance
                if support and resistance:
                    # Price near support - potential buy
                    if abs(current_price - support) / current_price < 0.02:
                        score += 0.1
                    # Price near resistance - potential sell
                    elif abs(current_price - resistance) / current_price < 0.02:
                        score -= 0.1
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical score: {e}")
        
        return max(0, min(1, score))
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi < 30:
            return "Oversold"
        elif rsi > 70:
            return "Overbought"
        else:
            return "Neutral"
    
    def _fallback_analysis(self, symbol: str, reason: str) -> AgentAnalysis:
        """Fallback analysis when technical data is unavailable"""
        return AgentAnalysis(
            agent_type=self.agent_type,
            confidence=0.1,
            signal="HOLD",
            reasoning=reason,
            data_sources=[],
            timestamp=datetime.now(),
            risk_score=0.9,
            metadata={}
        )


class LNMarketsRiskAgent(BaseAgent):
    """Risk management agent for LNMarkets trading"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.RISK_MANAGER, config)
        self.max_position_size = config.get('max_position_size', 0.02)
        self.max_daily_loss = config.get('max_daily_loss', 0.03)
        self.max_leverage = config.get('max_leverage', 5)
        self.lnmarkets_client = LNMarketsOfficialClient()
        
    async def analyze(self, symbol: str, market_data: Dict[str, Any]) -> AgentAnalysis:
        """Analyze risk for proposed trade on LNMarkets"""
        try:
            # Get current account state
            account_data = await self._get_account_data()
            proposed_trade = market_data.get('proposed_trade', {})
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(account_data, proposed_trade, market_data)
            
            # Generate risk assessment
            risk_score = risk_metrics['overall_risk']
            
            if risk_score < 0.3:
                signal = "APPROVE"
                confidence = 0.9
            elif risk_score < 0.6:
                signal = "CAUTION"
                confidence = 0.6
            else:
                signal = "REJECT"
                confidence = 0.9
            
            reasoning = f"Risk assessment: {risk_score:.2f}. "
            reasoning += f"Total balance: {account_data.get('balance', 0)} sats. "
            reasoning += f"Available balance: {account_data.get('available_balance', 0)} sats. "
            reasoning += f"Position risk: {risk_metrics.get('position_risk', 0):.2f}, "
            reasoning += f"Volatility risk: {risk_metrics.get('volatility_risk', 0):.2f}."
            
            return AgentAnalysis(
                agent_type=self.agent_type,
                confidence=confidence,
                signal=signal,
                reasoning=reasoning,
                data_sources=["LNMarkets Account", "Risk Analysis"],
                timestamp=datetime.now(),
                risk_score=risk_score,
                metadata={
                    'risk_metrics': risk_metrics,
                    'account_data': account_data,
                    'recommended_position_size': risk_metrics.get('recommended_position_size', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return self._fallback_analysis(symbol, "Risk analysis error")
    
    async def _get_account_data(self) -> Dict[str, Any]:
        """Get current account data from LNMarkets"""
        try:
            # Initialize client if needed
            if not self.lnmarkets_client.is_connected:
                self.lnmarkets_client.initialize()
            
            user_data = self.lnmarkets_client.get_user_info()
            positions = self.lnmarkets_client.get_positions()
            
            # Calculate total margin used in open positions
            total_margin_used = 0
            if isinstance(positions, list):
                for pos in positions:
                    total_margin_used += pos.get('margin', 0)
            
            total_balance = user_data.get('balance', 0)
            available_balance = total_balance - total_margin_used
            
            return {
                'balance': total_balance,
                'available_balance': available_balance,
                'margin_used': total_margin_used,
                'username': user_data.get('username', ''),
                'positions': positions if isinstance(positions, list) else [],
                'position_count': len(positions) if isinstance(positions, list) else 0
            }
        except Exception as e:
            self.logger.error(f"Account data fetch failed: {e}")
            return {
                'balance': 100000,  # Default fallback
                'available_balance': 100000,  # Default fallback
                'margin_used': 0,
                'positions': [],
                'position_count': 0
            }
    
    def _calculate_risk_metrics(self, account_data: Dict, proposed_trade: Dict, market_data: Dict) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for Bitcoin trading"""
        try:
            # Account metrics - use available balance for risk calculations
            current_balance = account_data.get('balance', 100000)
            available_balance = account_data.get('available_balance', current_balance)
            current_positions = account_data.get('positions', [])
            
            # Market metrics
            volatility = market_data.get('technical_analysis', {}).get('volatility', 0.02)
            current_price = market_data.get('market_data', {}).get('price', 50000)
            
            # Proposed trade metrics
            position_size = proposed_trade.get('position_size', 0)
            leverage = proposed_trade.get('leverage', 1)
            
            # Calculate position risk using available balance
            position_value = position_size * available_balance
            position_risk = position_value / available_balance
            
            # Calculate volatility risk
            volatility_risk = volatility * leverage  # Higher volatility + leverage = higher risk
            
            # Calculate concentration risk based on total balance
            total_exposure = sum(pos.get('quantity', 0) for pos in current_positions)
            concentration_risk = (total_exposure + position_value) / current_balance
            
            # Calculate leverage risk (adjusted for 10-33x range)
            leverage_risk = max(0, (leverage - 10) / 23)  # Risk increases above 10x
            
            # Overall risk score
            overall_risk = (
                position_risk * 0.3 + 
                volatility_risk * 0.25 + 
                concentration_risk * 0.25 + 
                leverage_risk * 0.2
            )
            
            # Recommended position size based on available balance
            max_safe_position = available_balance * self.max_position_size
            volatility_adjusted_position = max_safe_position * (1 - volatility)
            recommended_position_size = min(max_safe_position, volatility_adjusted_position)
            
            # Dynamic leverage calculation that respects configured max
            # Higher leverage for low volatility, lower for high volatility
            # Adjusted for 10-33x range
            base_leverage = 10  # Minimum leverage
            leverage_range = self.max_leverage - base_leverage  # 33 - 10 = 23
            volatility_adjusted_leverage = base_leverage + (leverage_range * (1 - volatility * 2)) if volatility > 0 else self.max_leverage
            max_safe_leverage = max(base_leverage, min(self.max_leverage, volatility_adjusted_leverage))
            
            return {
                'position_risk': position_risk,
                'volatility_risk': volatility_risk,
                'concentration_risk': concentration_risk,
                'leverage_risk': leverage_risk,
                'overall_risk': overall_risk,
                'recommended_position_size': recommended_position_size,
                'max_safe_leverage': max_safe_leverage
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {'overall_risk': 0.9}
    
    def _fallback_analysis(self, symbol: str, reason: str) -> AgentAnalysis:
        """Fallback analysis when risk data is unavailable"""
        return AgentAnalysis(
            agent_type=self.agent_type,
            confidence=0.9,
            signal="REJECT",
            reasoning=f"Risk analysis failed: {reason}",
            data_sources=[],
            timestamp=datetime.now(),
            risk_score=0.9,
            metadata={}
        )


class SelfHostedMultiAgentSystem:
    """Self-hosted multi-agent trading system for LNMarkets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self.logger = logging.getLogger("selfhosted_multi_agent_system")
        self.cache = TradingCache()
        self.data_provider = LNMarketsDataProvider()
        
        # Initialize agents
        self._initialize_agents()
        
        # Trading execution client
        self.trading_client = LNMarketsOfficialClient()
        # Initialize the client connection
        if not self.trading_client.initialize():
            self.logger.warning("Failed to initialize LNMarkets client on startup")
        
    def _initialize_agents(self):
        """Initialize all trading agents for self-hosted operation"""
        try:
            # Bitcoin-focused agents
            self.agents[AgentType.FUNDAMENTAL] = BitcoinFundamentalAgent(self.config)
            self.agents[AgentType.SENTIMENT] = CryptoSentimentAgent(self.config)
            self.agents[AgentType.TECHNICAL] = LNMarketsTechnicalAgent(self.config)
            self.agents[AgentType.RISK_MANAGER] = LNMarketsRiskAgent(self.config)
            
            self.logger.info("All self-hosted agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def analyze_symbol(self, symbol: str = "BTCUSD") -> List[AgentAnalysis]:
        """Get analysis from all agents for Bitcoin trading"""
        try:
            # Get comprehensive market data first
            market_data = await self.data_provider.get_comprehensive_data(symbol)
            
            if 'error' in market_data:
                self.logger.error(f"Market data error: {market_data['error']}")
                return []
            
            # Run agents in parallel for efficiency
            analyses = []
            tasks = []
            
            for agent_type, agent in self.agents.items():
                if agent_type != AgentType.RISK_MANAGER:  # Risk manager runs separately
                    task = asyncio.create_task(agent.analyze(symbol, market_data))
                    tasks.append(task)
            
            # Wait for all analyses to complete
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any exceptions
            for result in agent_results:
                if isinstance(result, AgentAnalysis):
                    analyses.append(result)
                else:
                    self.logger.error(f"Agent analysis failed: {result}")
            
            return analyses
            
        except Exception as e:
            self.logger.error(f"Symbol analysis failed: {e}")
            return []
    
    async def generate_trading_decision(self, symbol: str = "BTCUSD") -> TradingDecision:
        """Generate a trading decision based on multi-agent analysis"""
        try:
            # Get analyses from all agents
            analyses = await self.analyze_symbol(symbol)
            
            if not analyses:
                return self._no_decision(symbol, "No agent analyses available")
            
            # Calculate agent consensus
            consensus = self._calculate_consensus(analyses)
            
            # Get current market data for risk assessment
            market_data = await self.data_provider.get_comprehensive_data(symbol)
            
            # Prepare data for risk manager
            proposed_trade = {
                'symbol': symbol,
                'action': consensus['action'],
                'position_size': consensus['position_size'],
                'leverage': consensus.get('leverage', 2),
                'entry_price': market_data.get('market_data', {}).get('price', 0)
            }
            
            # Get risk assessment
            risk_data = market_data.copy()
            risk_data['proposed_trade'] = proposed_trade
            
            risk_analysis = await self.agents[AgentType.RISK_MANAGER].analyze(symbol, risk_data)
            
            # Final decision based on consensus and risk assessment
            if risk_analysis.signal == "REJECT":
                return self._no_decision(symbol, f"Risk manager rejected: {risk_analysis.reasoning}")
            elif risk_analysis.signal == "CAUTION":
                # Reduce position size and leverage for caution
                consensus['position_size'] *= 0.5
                consensus['leverage'] = min(consensus.get('leverage', 2), 2)
                consensus['confidence'] *= 0.8
            
            return TradingDecision(
                symbol=symbol,
                action=consensus['action'],
                confidence=consensus['confidence'],
                position_size=consensus['position_size'],
                stop_loss=consensus['stop_loss'],
                take_profit=consensus['take_profit'],
                reasoning=consensus['reasoning'],
                risk_assessment=risk_analysis.metadata,
                agent_consensus={
                    'total_agents': len(analyses),
                    'buy_votes': sum(1 for a in analyses if a.signal == "BUY"),
                    'sell_votes': sum(1 for a in analyses if a.signal == "SELL"),
                    'hold_votes': sum(1 for a in analyses if a.signal == "HOLD"),
                    'avg_confidence': sum(a.confidence for a in analyses) / len(analyses),
                    'risk_score': risk_analysis.risk_score
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Trading decision generation failed: {e}")
            return self._no_decision(symbol, f"Decision generation error: {e}")
    
    def _calculate_consensus(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """Calculate consensus from multiple agent analyses"""
        if not analyses:
            return {
                'action': 'HOLD',
                'confidence': 0.1,
                'position_size': 0,
                'leverage': 1,
                'stop_loss': None,
                'take_profit': None,
                'reasoning': 'No analyses available'
            }
        
        # Weight agents by confidence
        weighted_signals = []
        total_weight = 0
        
        for analysis in analyses:
            weight = analysis.confidence
            total_weight += weight
            
            if analysis.signal == "BUY":
                weighted_signals.append(weight)
            elif analysis.signal == "SELL":
                weighted_signals.append(-weight)
            else:  # HOLD
                weighted_signals.append(0)
        
        # Calculate weighted average
        if total_weight > 0:
            consensus_score = sum(weighted_signals) / total_weight
        else:
            consensus_score = 0
        
        # Determine action
        if consensus_score > 0.3:
            action = "BUY"
            confidence = min(0.95, abs(consensus_score) + 0.2)
        elif consensus_score < -0.3:
            action = "SELL"
            confidence = min(0.95, abs(consensus_score) + 0.2)
        else:
            action = "HOLD"
            confidence = 0.5
        
        # Calculate position size (percentage of account)
        # Use smaller base size for more reasonable USD amounts
        base_position_size = 0.002  # 0.2% base (much smaller)
        position_size = base_position_size * confidence  # Scale by confidence
        
        # Cap position size at max from config
        max_position = self.config.get('max_position_size', 0.03)
        position_size = min(position_size, max_position)
        
        # Determine leverage based on confidence and market conditions
        # Use 10-33x leverage range
        max_leverage = self.config.get('max_leverage', 33)
        min_leverage = 10
        leverage_range = max_leverage - min_leverage  # 23
        leverage = min(max_leverage, min_leverage + (confidence * leverage_range))  # 10x to 33x based on confidence
        
        # Generate reasoning
        buy_agents = [a.agent_type.value for a in analyses if a.signal == "BUY"]
        sell_agents = [a.agent_type.value for a in analyses if a.signal == "SELL"]
        hold_agents = [a.agent_type.value for a in analyses if a.signal == "HOLD"]
        
        reasoning = f"Self-hosted multi-agent consensus: {action} (score: {consensus_score:.2f}). "
        reasoning += f"BUY votes: {buy_agents}, SELL votes: {sell_agents}, HOLD votes: {hold_agents}."
        
        return {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'leverage': leverage,
            'stop_loss': None,  # Could be calculated based on technical analysis
            'take_profit': None,  # Could be calculated based on technical analysis
            'reasoning': reasoning
        }
    
    def _no_decision(self, symbol: str, reason: str) -> TradingDecision:
        """Generate a no-action decision"""
        return TradingDecision(
            symbol=symbol,
            action="HOLD",
            confidence=0.1,
            position_size=0,
            stop_loss=None,
            take_profit=None,
            reasoning=reason,
            risk_assessment={},
            agent_consensus={},
            timestamp=datetime.now()
        )
    
    async def _get_account_data(self) -> Dict[str, Any]:
        """Get current account data from LNMarkets"""
        try:
            # Initialize client if needed
            if not self.trading_client.is_connected:
                self.trading_client.initialize()
            
            user_data = self.trading_client.get_user_info()
            positions = self.trading_client.get_positions()
            
            # Calculate total margin used in open positions
            total_margin_used = 0
            if isinstance(positions, list):
                for pos in positions:
                    total_margin_used += pos.get('margin', 0)
            
            total_balance = user_data.get('balance', 0)
            available_balance = total_balance - total_margin_used
            
            return {
                'balance': total_balance,
                'available_balance': available_balance,
                'margin_used': total_margin_used,
                'username': user_data.get('username', ''),
                'positions': positions if isinstance(positions, list) else [],
                'position_count': len(positions) if isinstance(positions, list) else 0
            }
        except Exception as e:
            self.logger.error(f"Account data fetch failed: {e}")
            return {
                'balance': 100000,  # Default fallback
                'available_balance': 100000,  # Default fallback
                'margin_used': 0,
                'positions': [],
                'position_count': 0
            }
    
    async def execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute a trading decision on LNMarkets"""
        try:
            if decision.action == "HOLD" or decision.position_size == 0:
                return {
                    'status': 'no_action',
                    'reason': decision.reasoning,
                    'timestamp': datetime.now()
                }
            
            # Ensure client is connected
            if not self.trading_client.is_connected:
                if not self.trading_client.initialize():
                    return {
                        'status': 'failed',
                        'error': 'Failed to connect to LNMarkets',
                        'decision': decision,
                        'timestamp': datetime.now()
                    }
            
            # Get current account data to use available balance
            account_data = await self._get_account_data()
            available_balance = account_data.get('available_balance', 100000)
            
            # Execute trade through LNMarkets client
            side = 'long' if decision.action == "BUY" else 'short'
            
            # Calculate position parameters using available balance
            # Get leverage from decision or use configured max
            leverage = decision.risk_assessment.get('max_safe_leverage', self.config.get('max_leverage', 33))
            
            # Calculate margin in sats (position_size is a percentage)
            margin = decision.position_size * available_balance
            
            # Cap margin at reasonable USD equivalent (~100-200 USD max)
            # Assuming ~75k USD per BTC, 100 USD = ~133 sats, 200 USD = ~266 sats
            max_margin_sats = 300  # Approximately 200-250 USD worth
            margin = min(margin, max_margin_sats)
            
            self.logger.info(f"Trade params: margin={margin} sats, leverage={leverage}x, side={decision.action}")
            
            # Use the place_market_order method which properly handles the API call
            order_id = self.trading_client.place_market_order(
                side=side,
                quantity=int(margin),
                leverage=leverage,
                take_profit=decision.take_profit,
                stop_loss=decision.stop_loss
            )
            
            if order_id:
                return {
                    'status': 'executed',
                    'trade_id': order_id,
                    'side': side,
                    'margin': margin,
                    'leverage': leverage,
                    'decision': decision,
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Failed to place order',
                    'decision': decision,
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'decision': decision,
                'timestamp': datetime.now()
            }
    
    async def run_analysis_cycle(self, symbol: str = "BTCUSD") -> Dict[str, Any]:
        """Run a complete self-hosted analysis cycle"""
        try:
            self.logger.info(f"Starting self-hosted analysis cycle for {symbol}")
            
            # Generate trading decision
            decision = await self.generate_trading_decision(symbol)
            
            # Log decision
            self.logger.info(f"Trading decision for {symbol}: {decision.action} "
                           f"(confidence: {decision.confidence:.2f}, "
                           f"size: {decision.position_size:.4f})")
            
            return {
                'symbol': symbol,
                'decision': decision,
                'system_type': 'self_hosted',
                'data_sources': ['LNMarkets', 'Free APIs', 'Crypto News'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Self-hosted analysis cycle failed: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'system_type': 'self_hosted',
                'timestamp': datetime.now()
            }


if __name__ == "__main__":
    # Example usage
    config = {
        'max_position_size': 0.003,  # 0.3% max per position for smaller USD amounts
        'max_daily_loss': 0.03,      # 3% max daily loss
        'max_leverage': 33           # 33x max leverage as requested
    }
    
    async def main():
        system = SelfHostedMultiAgentSystem(config)
        result = await system.run_analysis_cycle("BTCUSD")
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())
