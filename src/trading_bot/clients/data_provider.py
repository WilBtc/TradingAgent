import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import requests

from redis_cache import TradingCache
from lnmarkets_official_client import LNMarketsOfficialClient
from real_technical_indicators import TechnicalIndicators


@dataclass
class LNMarketData:
    symbol: str
    price: float
    volume: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    funding_rate: Optional[float]
    open_interest: Optional[float]
    timestamp: datetime
    source: str = "LNMarkets"


@dataclass
class CryptoNewsItem:
    title: str
    content: str
    url: str
    published: datetime
    sentiment_score: float
    source: str
    relevance_score: float
    bitcoin_relevance: float


@dataclass
class BitcoinFundamentals:
    symbol: str
    market_cap: Optional[float]
    circulating_supply: Optional[float]
    max_supply: Optional[float]
    hash_rate: Optional[float]
    difficulty: Optional[float]
    active_addresses: Optional[int]
    transaction_count: Optional[int]
    network_value: Optional[float]
    fear_greed_index: Optional[int]
    dominance_percentage: Optional[float]
    timestamp: datetime


class LNMarketsDataProvider:
    """Self-hosted data provider using only LNMarkets and free APIs"""
    
    def __init__(self):
        self.cache = TradingCache()
        self.logger = logging.getLogger("lnmarkets_data_provider")
        self.lnmarkets_client = LNMarketsOfficialClient()
        self.technical_indicators = TechnicalIndicators()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    async def get_market_data(self, symbol: str = "BTCUSD") -> LNMarketData:
        """Get real-time market data from LNMarkets"""
        cache_key = f"lnmarkets_ticker_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            # Convert timestamp string back to datetime
            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return LNMarketData(**data)
        
        try:
            # Initialize client and get ticker data from LNMarkets
            if not self.lnmarkets_client.is_connected:
                self.lnmarkets_client.initialize()
            
            ticker_data = self.lnmarkets_client.get_market_data()
            
            if ticker_data:
                market_data = LNMarketData(
                    symbol=symbol,
                    price=float(ticker_data.get('price', 0)),
                    volume=float(ticker_data.get('volume_24h', 0)),
                    change_24h=float(ticker_data.get('change_24h', 0)),
                    change_percent_24h=float(ticker_data.get('change_percent_24h', 0)),
                    high_24h=float(ticker_data.get('high_24h', 0)),
                    low_24h=float(ticker_data.get('low_24h', 0)),
                    funding_rate=ticker_data.get('funding_rate'),
                    open_interest=ticker_data.get('open_interest'),
                    timestamp=datetime.now()
                )
                
                # Cache for 30 seconds
                self.cache.set(
                    cache_key,
                    json.dumps(market_data.__dict__, default=str),
                    ttl=30
                )
                
                return market_data
            else:
                raise Exception("No ticker data received")
                
        except Exception as e:
            self.logger.error(f"LNMarkets market data fetch failed: {e}")
            # Return fallback data
            return LNMarketData(
                symbol=symbol,
                price=0.0,
                volume=0.0,
                change_24h=0.0,
                change_percent_24h=0.0,
                high_24h=0.0,
                low_24h=0.0,
                funding_rate=None,
                open_interest=None,
                timestamp=datetime.now()
            )
    
    async def get_ticker(self, symbol: str = "BTCUSD") -> Dict[str, Any]:
        """Get current ticker data in a simplified format"""
        try:
            # Get market data using existing method
            market_data = await self.get_market_data(symbol)
            
            if market_data:
                # Convert to ticker format expected by profit_maximizer
                return {
                    'last': market_data.price,
                    'bid': market_data.price - 10,  # Approximate bid (slightly below last)
                    'ask': market_data.price + 10,  # Approximate ask (slightly above last)
                    'price': market_data.price,
                    'volume': market_data.volume,
                    'high': market_data.high_24h,
                    'low': market_data.low_24h,
                    'change_24h': market_data.change_24h,
                    'change_percent_24h': market_data.change_percent_24h,
                    'timestamp': market_data.timestamp
                }
            else:
                # Return empty ticker if no data
                return {
                    'last': 0.0,
                    'bid': 0.0,
                    'ask': 0.0,
                    'price': 0.0,
                    'volume': 0.0,
                    'high': 0.0,
                    'low': 0.0,
                    'change_24h': 0.0,
                    'change_percent_24h': 0.0,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Get ticker failed: {e}")
            # Return fallback ticker data
            return {
                'last': 0.0,
                'bid': 0.0,
                'ask': 0.0,
                'price': 0.0,
                'volume': 0.0,
                'high': 0.0,
                'low': 0.0,
                'change_24h': 0.0,
                'change_percent_24h': 0.0,
                'timestamp': datetime.now()
            }
    
    async def get_historical_data(self, symbol: str = "BTCUSD", limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical candle data from LNMarkets"""
        cache_key = f"lnmarkets_candles_{symbol}_{limit}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        try:
            # Initialize client if needed
            if not self.lnmarkets_client.is_connected:
                self.lnmarkets_client.initialize()
            
            # LNMarkets doesn't have a get_candlesticks method - use fallback data
            candles = []
            
            if candles:
                # Convert to standard format
                formatted_candles = []
                for candle in candles:
                    formatted_candles.append({
                        'timestamp': candle.get('timestamp'),
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0))
                    })
                
                # Cache for 5 minutes
                self.cache.set(
                    cache_key,
                    json.dumps(formatted_candles, default=str),
                    ttl=300
                )
                
                return formatted_candles
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Historical data fetch failed: {e}")
            return []
    
    async def get_bitcoin_fundamentals(self) -> BitcoinFundamentals:
        """Get Bitcoin fundamental data from free APIs"""
        cache_key = "bitcoin_fundamentals"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            # Convert timestamp string back to datetime
            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return BitcoinFundamentals(**data)
        
        try:
            fundamentals = BitcoinFundamentals(
                symbol="BTC",
                market_cap=None,
                circulating_supply=None,
                max_supply=21000000.0,  # Bitcoin fixed supply
                hash_rate=None,
                difficulty=None,
                active_addresses=None,
                transaction_count=None,
                network_value=None,
                fear_greed_index=None,
                dominance_percentage=None,
                timestamp=datetime.now()
            )
            
            # Try to get data from free APIs
            try:
                # CoinGecko API (free tier)
                coingecko_url = "https://api.coingecko.com/api/v3/coins/bitcoin"
                response = requests.get(coingecko_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    market_data = data.get('market_data', {})
                    
                    fundamentals.market_cap = market_data.get('market_cap', {}).get('usd')
                    fundamentals.circulating_supply = market_data.get('circulating_supply')
                    fundamentals.dominance_percentage = market_data.get('market_cap_rank')
                    
            except Exception as e:
                self.logger.warning(f"CoinGecko API failed: {e}")
            
            # Try Fear & Greed Index
            try:
                fg_url = "https://api.alternative.me/fng/"
                response = requests.get(fg_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data'):
                        fundamentals.fear_greed_index = int(data['data'][0]['value'])
                        
            except Exception as e:
                self.logger.warning(f"Fear & Greed API failed: {e}")
            
            # Cache for 1 hour
            self.cache.set(
                cache_key,
                json.dumps(fundamentals.__dict__, default=str),
                ttl=3600
            )
            
            return fundamentals
            
        except Exception as e:
            self.logger.error(f"Bitcoin fundamentals fetch failed: {e}")
            return BitcoinFundamentals(
                symbol="BTC",
                market_cap=None,
                circulating_supply=None,
                max_supply=21000000.0,
                hash_rate=None,
                difficulty=None,
                active_addresses=None,
                transaction_count=None,
                network_value=None,
                fear_greed_index=50,  # Neutral
                dominance_percentage=None,
                timestamp=datetime.now()
            )
    
    async def get_crypto_news(self, hours: int = 24) -> List[CryptoNewsItem]:
        """Get crypto news from free RSS feeds"""
        cache_key = f"crypto_news_{hours}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            # Convert published string back to datetime
            for item in data:
                if isinstance(item.get('published'), str):
                    item['published'] = datetime.fromisoformat(item['published'])
            return [CryptoNewsItem(**item) for item in data]
        
        news_items = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Free crypto news sources
        rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/"
        ]
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Limit per feed
                    # Parse published date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published = datetime(*entry.published_parsed[:6])
                        else:
                            published = datetime.now()
                    except:
                        published = datetime.now()
                    
                    if published < cutoff_time:
                        continue
                    
                    # Analyze content
                    text = f"{entry.title} {entry.get('summary', '')}"
                    
                    # Calculate Bitcoin relevance
                    bitcoin_relevance = self._calculate_bitcoin_relevance(text)
                    
                    if bitcoin_relevance < 0.1:  # Skip non-Bitcoin news
                        continue
                    
                    # Analyze sentiment
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    
                    news_item = CryptoNewsItem(
                        title=entry.title,
                        content=entry.get('summary', ''),
                        url=entry.link,
                        published=published,
                        sentiment_score=sentiment['compound'],
                        source=feed_url.split('/')[2],  # Extract domain
                        relevance_score=self._calculate_relevance_score(text),
                        bitcoin_relevance=bitcoin_relevance
                    )
                    news_items.append(news_item)
                    
            except Exception as e:
                self.logger.error(f"RSS feed {feed_url} failed: {e}")
        
        # Sort by Bitcoin relevance and recency
        news_items.sort(key=lambda x: (x.bitcoin_relevance, x.published), reverse=True)
        
        # Cache for 15 minutes
        self.cache.set(
            cache_key,
            json.dumps([item.__dict__ for item in news_items[:20]], default=str),
            ttl=900
        )
        
        return news_items[:20]
    
    def _calculate_bitcoin_relevance(self, text: str) -> float:
        """Calculate how relevant the text is to Bitcoin"""
        text_lower = text.lower()
        relevance = 0.0
        
        # Bitcoin keywords
        bitcoin_keywords = {
            'bitcoin': 0.8,
            'btc': 0.7,
            'bitcoin price': 0.9,
            'bitcoin market': 0.8,
            'bitcoin trading': 0.9,
            'satoshi': 0.6,
            'blockchain': 0.4,
            'cryptocurrency': 0.3,
            'crypto': 0.3,
            'digital currency': 0.4,
            'halving': 0.7,
            'mining': 0.5,
            'hash rate': 0.6,
            'lightning network': 0.8
        }
        
        for keyword, weight in bitcoin_keywords.items():
            if keyword in text_lower:
                relevance += weight
        
        return min(1.0, relevance)
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate general relevance score for trading"""
        text_lower = text.lower()
        relevance = 0.0
        
        # Trading-relevant keywords
        trading_keywords = [
            'price', 'trading', 'market', 'volume', 'support', 'resistance',
            'bullish', 'bearish', 'rally', 'dump', 'pump', 'breakout',
            'technical analysis', 'fundamental', 'adoption', 'institutional',
            'regulation', 'etf', 'futures', 'options'
        ]
        
        for keyword in trading_keywords:
            if keyword in text_lower:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    async def get_technical_analysis(self, symbol: str = "BTCUSD") -> Dict[str, Any]:
        """Get technical analysis using existing indicators"""
        cache_key = f"technical_analysis_{symbol}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        try:
            # Get historical data
            candles = await self.get_historical_data(symbol, limit=200)
            
            if len(candles) < 50:
                return {'error': 'Insufficient data for technical analysis'}
            
            # Extract price arrays
            closes = [candle['close'] for candle in candles]
            highs = [candle['high'] for candle in candles]
            lows = [candle['low'] for candle in candles]
            volumes = [candle['volume'] for candle in candles]
            
            # Calculate technical indicators
            analysis = {
                'current_price': closes[-1],
                'rsi': self.technical_indicators.calculate_rsi(closes, 14)[-1] if closes else None,
                'sma_20': self.technical_indicators.calculate_sma(closes, 20)[-1] if closes else None,
                'ema_12': self.technical_indicators.calculate_ema(closes, 12)[-1] if closes else None,
                'support_resistance': self.technical_indicators.find_support_resistance(closes, highs, lows),
                'trend': self._determine_trend(closes),
                'volatility': self._calculate_volatility(closes),
                'volume_trend': self._analyze_volume_trend(volumes),
                'timestamp': datetime.now()
            }
            
            # MACD
            try:
                macd_line, macd_signal, macd_histogram = self.technical_indicators.calculate_macd(closes)
                analysis.update({
                    'macd_line': macd_line[-1] if macd_line else None,
                    'macd_signal': macd_signal[-1] if macd_signal else None,
                    'macd_histogram': macd_histogram[-1] if macd_histogram else None
                })
            except Exception as e:
                self.logger.warning(f"MACD calculation failed: {e}")
            
            # Cache for 2 minutes
            self.cache.set(
                cache_key,
                json.dumps(analysis, default=str),
                ttl=120
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {'error': str(e)}
    
    def _determine_trend(self, closes: List[float]) -> str:
        """Determine the current trend"""
        if len(closes) < 20:
            return 'neutral'
        
        recent_20 = closes[-20:]
        first_10_avg = sum(recent_20[:10]) / 10
        last_10_avg = sum(recent_20[-10:]) / 10
        
        change_percent = ((last_10_avg - first_10_avg) / first_10_avg) * 100
        
        if change_percent > 2:
            return 'bullish'
        elif change_percent < -2:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_volatility(self, closes: List[float]) -> float:
        """Calculate price volatility"""
        if len(closes) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(closes)):
            returns.append((closes[i] - closes[i-1]) / closes[i-1])
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return variance ** 0.5
    
    def _analyze_volume_trend(self, volumes: List[float]) -> str:
        """Analyze volume trend"""
        if len(volumes) < 10:
            return 'neutral'
        
        recent_5 = volumes[-5:]
        previous_5 = volumes[-10:-5]
        
        recent_avg = sum(recent_5) / len(recent_5)
        previous_avg = sum(previous_5) / len(previous_5)
        
        if recent_avg > previous_avg * 1.2:
            return 'increasing'
        elif recent_avg < previous_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    async def get_comprehensive_data(self, symbol: str = "BTCUSD") -> Dict[str, Any]:
        """Get all data in one comprehensive call"""
        try:
            # Gather all data in parallel
            market_data_task = self.get_market_data(symbol)
            fundamentals_task = self.get_bitcoin_fundamentals()
            news_task = self.get_crypto_news(24)
            technical_task = self.get_technical_analysis(symbol)
            
            market_data, fundamentals, news_data, technical_data = await asyncio.gather(
                market_data_task, fundamentals_task, news_task, technical_task,
                return_exceptions=True
            )
            
            # Calculate aggregated sentiment
            news_sentiment = self._calculate_news_sentiment(news_data if isinstance(news_data, list) else [])
            
            return {
                'symbol': symbol,
                'market_data': market_data.__dict__ if hasattr(market_data, '__dict__') else market_data,
                'fundamentals': fundamentals.__dict__ if hasattr(fundamentals, '__dict__') else fundamentals,
                'news_data': [item.__dict__ for item in news_data[:10]] if isinstance(news_data, list) else [],
                'technical_analysis': technical_data if isinstance(technical_data, dict) else {},
                'sentiment_analysis': {
                    'news_sentiment': news_sentiment,
                    'fear_greed_index': fundamentals.fear_greed_index if hasattr(fundamentals, 'fear_greed_index') else 50,
                    'overall_sentiment': self._calculate_overall_sentiment(news_sentiment, fundamentals)
                },
                'data_quality': self._assess_data_quality(market_data, fundamentals, news_data, technical_data),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive data gathering failed: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _calculate_news_sentiment(self, news_items: List[CryptoNewsItem]) -> float:
        """Calculate aggregated news sentiment"""
        if not news_items:
            return 0.0
        
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        now = datetime.now()
        
        for item in news_items:
            # Calculate time decay
            hours_old = (now - item.published).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_old / 24))
            
            # Weight by Bitcoin relevance and time
            total_item_weight = item.bitcoin_relevance * time_weight
            
            weighted_sentiment += item.sentiment_score * total_item_weight
            total_weight += total_item_weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_overall_sentiment(self, news_sentiment: float, fundamentals: Any) -> float:
        """Calculate overall market sentiment"""
        sentiment_score = news_sentiment * 0.6  # News weight
        
        # Add Fear & Greed Index
        if hasattr(fundamentals, 'fear_greed_index') and fundamentals.fear_greed_index is not None:
            # Convert 0-100 to -1 to 1
            fg_normalized = (fundamentals.fear_greed_index - 50) / 50
            sentiment_score += fg_normalized * 0.4  # F&G weight
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _assess_data_quality(self, market_data: Any, fundamentals: Any, news_data: Any, technical_data: Any) -> Dict[str, float]:
        """Assess the quality of gathered data"""
        quality_scores = {
            'market_data_quality': 1.0 if hasattr(market_data, 'price') and market_data.price > 0 else 0.0,
            'fundamentals_quality': 0.8 if hasattr(fundamentals, 'market_cap') else 0.5,
            'news_data_quality': min(1.0, len(news_data) / 5) if isinstance(news_data, list) else 0.0,
            'technical_data_quality': 1.0 if isinstance(technical_data, dict) and 'rsi' in technical_data else 0.0
        }
        
        quality_scores['overall_quality'] = sum(quality_scores.values()) / len(quality_scores)
        
        return quality_scores


if __name__ == "__main__":
    async def main():
        provider = LNMarketsDataProvider()
        data = await provider.get_comprehensive_data("BTCUSD")
        print(json.dumps(data, indent=2, default=str))
    
    asyncio.run(main())
