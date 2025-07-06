#!/usr/bin/env python3
"""
DeepSeek LLM Integration - Cheap AI Analysis for Trading
Using DeepSeek as a cost-effective alternative to GPT-4
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysis:
    """AI-generated market analysis"""
    sentiment: str  # bullish, bearish, neutral
    confidence: float  # 0-1
    reasoning: str
    trade_recommendation: str  # buy, sell, hold
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_level: str = "medium"
    time_horizon: str = "short"  # short, medium, long

@dataclass
class TradingSignal:
    """AI-generated trading signal"""
    action: str  # buy, sell, hold
    confidence: float
    reasoning: str
    urgency: str  # low, medium, high
    expected_move: float  # percentage
    risk_reward_ratio: float

class DeepSeekTradingAI:
    """DeepSeek AI integration for trading analysis"""
    
    def __init__(self, api_key: str = None):
        # DeepSeek API configuration
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or 'sk-c421e1af764e4ff48c726a0b4cdcec48'
        self.client = None
        
        # Only initialize client if API key is available
        if self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info("🤖 DeepSeek AI Trading Assistant initialized with API")
            except Exception as e:
                logger.warning(f"DeepSeek API initialization failed: {e}")
                self.client = None
        else:
            logger.info("🤖 DeepSeek AI initialized in fallback mode (no API key)")
        
        # Model configuration
        self.model = "deepseek-chat"  # Much cheaper than GPT-4
        self.max_tokens = 1000
        self.temperature = 0.1  # Low temperature for consistent analysis
    
    async def analyze_market_conditions(self, market_data: Dict, indicators: Dict) -> MarketAnalysis:
        """Analyze market conditions using DeepSeek AI"""
        try:
            # Use AI if available, otherwise fallback
            if self.client:
                prompt = self._create_market_analysis_prompt(market_data, indicators)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_market_analyst_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                analysis_text = response.choices[0].message.content
                return self._parse_market_analysis(analysis_text)
            else:
                logger.info("Using fallback analysis (no AI API)")
                return self._fallback_market_analysis(market_data, indicators)
            
        except Exception as e:
            logger.error(f"❌ DeepSeek market analysis error: {e}")
            return self._fallback_market_analysis(market_data, indicators)
    
    async def generate_trading_signal(self, market_data: Dict, indicators: Dict, 
                                    portfolio_data: Dict = None) -> TradingSignal:
        """Generate trading signal using DeepSeek AI"""
        try:
            if self.client:
                prompt = self._create_trading_signal_prompt(market_data, indicators, portfolio_data)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_signal_generator_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                signal_text = response.choices[0].message.content
                return self._parse_trading_signal(signal_text)
            else:
                logger.info("Using fallback signal generation (no AI API)")
                return self._fallback_trading_signal(indicators)
            
        except Exception as e:
            logger.error(f"❌ DeepSeek signal generation error: {e}")
            return self._fallback_trading_signal(indicators)
    
    async def analyze_risk(self, trade_data: Dict, market_conditions: Dict) -> Dict[str, Any]:
        """Analyze trading risk using DeepSeek AI"""
        try:
            if not self.client:
                return {"risk_level": "medium", "analysis": "Risk analysis using fallback method"}
                
            prompt = f"""
            Analyze the risk for this trade setup:
            
            Trade: {trade_data.get('action', 'N/A')} {trade_data.get('symbol', 'BTC')}
            Position Size: ${trade_data.get('quantity', 0):,.2f}
            Leverage: {trade_data.get('leverage', 1)}x
            Current Price: ${market_conditions.get('price', 0):,.2f}
            
            Market Conditions:
            - Volatility: {market_conditions.get('volatility', 'normal')}
            - Trend: {market_conditions.get('trend', 'neutral')}
            - Volume: {market_conditions.get('volume', 'normal')}
            
            Provide risk analysis with:
            1. Risk level (low/medium/high)
            2. Key risk factors
            3. Risk mitigation suggestions
            4. Maximum recommended position size
            5. Stop loss recommendation
            
            Respond in JSON format.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional risk management analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            risk_text = response.choices[0].message.content
            return self._parse_risk_analysis(risk_text)
            
        except Exception as e:
            logger.error(f"❌ DeepSeek risk analysis error: {e}")
            return {"risk_level": "medium", "warning": "AI analysis unavailable"}
    
    def _get_market_analyst_system_prompt(self) -> str:
        """System prompt for market analysis"""
        return """
        You are a professional cryptocurrency market analyst with expertise in technical analysis and market sentiment.
        
        Your role:
        - Analyze market data and technical indicators
        - Provide clear, actionable insights
        - Consider both technical and fundamental factors
        - Be conservative with recommendations
        - Always include risk considerations
        
        Response format:
        - Sentiment: bullish/bearish/neutral
        - Confidence: 0.0 to 1.0
        - Reasoning: Clear explanation
        - Recommendation: buy/sell/hold
        - Risk level: low/medium/high
        
        Keep responses concise and data-driven.
        """
    
    def _get_signal_generator_system_prompt(self) -> str:
        """System prompt for signal generation"""
        return """
        You are a professional trading signal generator specializing in cryptocurrency markets.
        
        Your role:
        - Generate precise trading signals
        - Consider multiple timeframes
        - Assess risk-reward ratios
        - Provide clear entry/exit criteria
        - Factor in market volatility
        
        Signal format:
        - Action: buy/sell/hold
        - Confidence: 0.0 to 1.0
        - Reasoning: Technical justification
        - Urgency: low/medium/high
        - Expected move: percentage
        
        Focus on high-probability setups only.
        """
    
    def _create_market_analysis_prompt(self, market_data: Dict, indicators: Dict) -> str:
        """Create market analysis prompt"""
        return f"""
        Analyze the current Bitcoin market conditions:
        
        PRICE DATA:
        - Current Price: ${market_data.get('price', 0):,.2f}
        - 24h Change: {market_data.get('change_24h', 0):.2f}%
        - Volume: {market_data.get('volume', 0):,.0f}
        
        TECHNICAL INDICATORS:
        - RSI (14): {indicators.get('rsi', 50):.2f}
        - MACD: {indicators.get('macd', 0):.4f}
        - MACD Signal: {indicators.get('macd_signal', 0):.4f}
        - SMA 20: ${indicators.get('sma_20', 0):,.2f}
        - SMA 50: ${indicators.get('sma_50', 0):,.2f}
        - Bollinger Bands: ${indicators.get('bb_lower', 0):,.2f} - ${indicators.get('bb_upper', 0):,.2f}
        - ATR: {indicators.get('atr', 0):.2f}
        - Trend Strength: {indicators.get('trend_strength', 0):.2f}
        - Volatility Regime: {indicators.get('volatility_regime', 'normal')}
        
        Provide comprehensive market analysis including sentiment, key levels, and trading recommendation.
        """
    
    def _create_trading_signal_prompt(self, market_data: Dict, indicators: Dict, 
                                    portfolio_data: Dict = None) -> str:
        """Create trading signal prompt"""
        portfolio_info = ""
        if portfolio_data:
            portfolio_info = f"""
            PORTFOLIO CONTEXT:
            - Current Positions: {portfolio_data.get('position_count', 0)}
            - Total P&L: ${portfolio_data.get('total_pnl', 0):.2f}
            - Available Balance: ${portfolio_data.get('available_balance', 0):.2f}
            """
        
        return f"""
        Generate a trading signal for Bitcoin based on this data:
        
        MARKET DATA:
        - Price: ${market_data.get('price', 0):,.2f}
        - Volume: {market_data.get('volume', 0):,.0f}
        - Volatility: {indicators.get('volatility_regime', 'normal')}
        
        TECHNICAL SETUP:
        - RSI: {indicators.get('rsi', 50):.2f}
        - MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):.4f})
        - Price vs SMA20: {((market_data.get('price', 0) / indicators.get('sma_20', 1)) - 1) * 100:.2f}%
        - Trend Strength: {indicators.get('trend_strength', 0):.2f}
        
        {portfolio_info}
        
        Generate a precise trading signal with confidence level and reasoning.
        """
    
    def _parse_market_analysis(self, analysis_text: str) -> MarketAnalysis:
        """Parse AI response into MarketAnalysis object"""
        try:
            # Try to extract JSON if present
            if "{" in analysis_text and "}" in analysis_text:
                start = analysis_text.find("{")
                end = analysis_text.rfind("}") + 1
                json_str = analysis_text[start:end]
                data = json.loads(json_str)
                
                return MarketAnalysis(
                    sentiment=data.get('sentiment', 'neutral'),
                    confidence=data.get('confidence', 0.5),
                    reasoning=data.get('reasoning', analysis_text[:200]),
                    trade_recommendation=data.get('recommendation', 'hold'),
                    price_target=data.get('price_target'),
                    stop_loss=data.get('stop_loss'),
                    risk_level=data.get('risk_level', 'medium')
                )
            
            # Fallback to text parsing
            sentiment = "neutral"
            if "bullish" in analysis_text.lower():
                sentiment = "bullish"
            elif "bearish" in analysis_text.lower():
                sentiment = "bearish"
            
            confidence = 0.5
            if "high confidence" in analysis_text.lower():
                confidence = 0.8
            elif "low confidence" in analysis_text.lower():
                confidence = 0.3
            
            recommendation = "hold"
            if "buy" in analysis_text.lower():
                recommendation = "buy"
            elif "sell" in analysis_text.lower():
                recommendation = "sell"
            
            return MarketAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                reasoning=analysis_text[:300],
                trade_recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error parsing market analysis: {e}")
            return MarketAnalysis(
                sentiment="neutral",
                confidence=0.5,
                reasoning=analysis_text[:200] if analysis_text else "Analysis unavailable",
                trade_recommendation="hold"
            )
    
    def _parse_trading_signal(self, signal_text: str) -> TradingSignal:
        """Parse AI response into TradingSignal object"""
        try:
            # Similar parsing logic for trading signals
            action = "hold"
            if "buy" in signal_text.lower():
                action = "buy"
            elif "sell" in signal_text.lower():
                action = "sell"
            
            confidence = 0.5
            if "high confidence" in signal_text.lower():
                confidence = 0.8
            elif "strong" in signal_text.lower():
                confidence = 0.7
            elif "weak" in signal_text.lower():
                confidence = 0.3
            
            return TradingSignal(
                action=action,
                confidence=confidence,
                reasoning=signal_text[:200],
                urgency="medium",
                expected_move=2.0,  # Default 2%
                risk_reward_ratio=2.0  # Default 1:2
            )
            
        except Exception as e:
            logger.error(f"Error parsing trading signal: {e}")
            return TradingSignal(
                action="hold",
                confidence=0.5,
                reasoning="Signal parsing failed",
                urgency="low",
                expected_move=0.0,
                risk_reward_ratio=1.0
            )
    
    def _parse_risk_analysis(self, risk_text: str) -> Dict[str, Any]:
        """Parse risk analysis from AI response"""
        try:
            # Try JSON parsing first
            if "{" in risk_text:
                start = risk_text.find("{")
                end = risk_text.rfind("}") + 1
                json_str = risk_text[start:end]
                return json.loads(json_str)
            
            # Fallback parsing
            risk_level = "medium"
            if "high risk" in risk_text.lower():
                risk_level = "high"
            elif "low risk" in risk_text.lower():
                risk_level = "low"
            
            return {
                "risk_level": risk_level,
                "analysis": risk_text[:300],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception:
            return {"risk_level": "medium", "error": "Risk analysis parsing failed"}
    
    def _fallback_market_analysis(self, market_data: Dict, indicators: Dict) -> MarketAnalysis:
        """Fallback analysis when AI is unavailable"""
        rsi = indicators.get('rsi', 50)
        trend_strength = indicators.get('trend_strength', 0)
        
        if rsi > 70 and trend_strength > 0.5:
            sentiment = "bearish"
            recommendation = "sell"
            confidence = 0.6
            reasoning = f"Overbought RSI ({rsi:.1f}) with strong uptrend may signal correction"
        elif rsi < 30 and trend_strength < -0.5:
            sentiment = "bullish"  
            recommendation = "buy"
            confidence = 0.6
            reasoning = f"Oversold RSI ({rsi:.1f}) with downtrend may signal bounce"
        else:
            sentiment = "neutral"
            recommendation = "hold"
            confidence = 0.4
            reasoning = "Mixed signals, no clear direction"
        
        return MarketAnalysis(
            sentiment=sentiment,
            confidence=confidence,
            reasoning=reasoning,
            trade_recommendation=recommendation
        )
    
    def _fallback_trading_signal(self, indicators: Dict) -> TradingSignal:
        """Fallback signal when AI is unavailable"""
        rsi = indicators.get('rsi', 50)
        
        if rsi > 70:
            return TradingSignal(
                action="sell",
                confidence=0.6,
                reasoning=f"RSI overbought at {rsi:.1f}",
                urgency="medium",
                expected_move=-3.0,
                risk_reward_ratio=1.5
            )
        elif rsi < 30:
            return TradingSignal(
                action="buy",
                confidence=0.6,
                reasoning=f"RSI oversold at {rsi:.1f}",
                urgency="medium", 
                expected_move=3.0,
                risk_reward_ratio=2.0
            )
        else:
            return TradingSignal(
                action="hold",
                confidence=0.5,
                reasoning="No clear signals",
                urgency="low",
                expected_move=0.0,
                risk_reward_ratio=1.0
            )

# Global instance
deepseek_ai = DeepSeekTradingAI()

async def test_deepseek_integration():
    """Test DeepSeek AI integration"""
    print("🤖 Testing DeepSeek AI Integration...")
    
    # Test data
    market_data = {
        'price': 97500,
        'volume': 25000,
        'change_24h': 2.5
    }
    
    indicators = {
        'rsi': 72.5,
        'macd': 150.0,
        'macd_signal': 120.0,
        'sma_20': 96000,
        'trend_strength': 0.7,
        'volatility_regime': 'normal'
    }
    
    ai = DeepSeekTradingAI()
    
    # Test market analysis
    print("📊 Testing market analysis...")
    analysis = await ai.analyze_market_conditions(market_data, indicators)
    print(f"   Sentiment: {analysis.sentiment}")
    print(f"   Confidence: {analysis.confidence:.2f}")
    print(f"   Recommendation: {analysis.trade_recommendation}")
    print(f"   Reasoning: {analysis.reasoning[:100]}...")
    
    # Test signal generation
    print("📡 Testing signal generation...")
    signal = await ai.generate_trading_signal(market_data, indicators)
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Expected Move: {signal.expected_move:.1f}%")
    print(f"   Reasoning: {signal.reasoning[:100]}...")
    
    # Test risk analysis
    print("⚠️ Testing risk analysis...")
    trade_data = {'action': 'buy', 'quantity': 1000, 'leverage': 2}
    risk = await ai.analyze_risk(trade_data, market_data)
    print(f"   Risk Level: {risk.get('risk_level')}")
    print(f"   Analysis: {str(risk)[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_deepseek_integration())