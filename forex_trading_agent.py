import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import schedule
import time
from threading import Thread
from openai import OpenAI
import ta
import pytz
import os
from dotenv import load_dotenv

# For local development only. Render will use real env vars.
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_trading_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexTradingAgent:
    def __init__(self):
        # API Keys (read from environment)
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openai_api_key     = os.getenv("OPENAI_API_KEY")
        self.twelve_data_api_key= os.getenv("TWELVE_DATA_API_KEY")

        # Initialize OpenAI client (using env key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Major Forex pairs - focus on EUR/USD and USD/JPY
        self.symbols = [
            'EUR/USD',   # Euro/US Dollar - most liquid forex pair
            'USD/JPY'    # US Dollar/Japanese Yen - popular carry trade pair
        ]
        
        # Set up Philippine timezone and major trading sessions
        self.ph_tz = pytz.timezone('Asia/Manila')
        self.london_tz = pytz.timezone('Europe/London')
        self.ny_tz = pytz.timezone('America/New_York')
        self.tokyo_tz = pytz.timezone('Asia/Tokyo')
        
        self.chat_id = None
        
        # Forex-specific technical indicators thresholds (more conservative than crypto)
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_threshold = 1.3  # 130% of average volume (forex has lower volatility)
        
        # Forex risk management (more conservative)
        self.risk_reward_ratio = 2.0  # 1:2 risk reward (standard for forex)
        self.max_risk_per_trade = 0.02  # 2% risk per trade for forex
        
        # Forex volatility multipliers for stop losses (much smaller than crypto)
        self.volatility_multipliers = {
            'EUR/USD': 1.0,    # Major pair - baseline
            'USD/JPY': 1.2     # JPY pairs tend to be slightly more volatile
        }
        
        # Forex trading sessions (PH time)
        self.trading_sessions = {
            'tokyo': {'start': '07:00', 'end': '16:00'},      # Tokyo session
            'london': {'start': '15:00', 'end': '24:00'},     # London session  
            'new_york': {'start': '21:00', 'end': '06:00'},   # New York session
            'overlap_london_ny': {'start': '21:00', 'end': '24:00'}  # High volatility overlap
        }

        # Store the main event loop reference
        self.main_loop = None

    async def get_forex_data(self, symbol: str, interval: str = '1h', outputsize: int = 100) -> Optional[pd.DataFrame]:
        """Fetch forex data from TwelveData API"""
        try:
            # TwelveData expects symbol format like EUR/USD
            api_symbol = symbol
            
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': api_symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.twelve_data_api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'values' in data and data['values']:
                            df = pd.DataFrame(data['values'])
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df.set_index('datetime', inplace=True)
                            
                            # Convert to numeric
                            numeric_columns = ['open', 'high', 'low', 'close']
                            
                            for col in numeric_columns:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Sort by datetime ascending
                            df = df.sort_index()
                            
                            logger.info(f"Successfully fetched data for {symbol}")
                            return df
                        else:
                            logger.warning(f"No data returned for {symbol}")
                            return None
                    else:
                        logger.error(f"API error for {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_forex_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators optimized for forex trading"""
        try:
            # RSI (14 period standard for forex)
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD (12, 26, 9) - excellent for forex trend following
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands (20, 2) - great for forex range detection
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Moving Averages - forex standard settings
            df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            
            # Stochastic Oscillator - excellent for forex
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ATR for stop loss calculation (14 period)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Williams %R - good for forex overbought/oversold
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Commodity Channel Index - trend strength
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            # Parabolic SAR - trend following
            df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
            
            # Awesome Oscillator - momentum
            df['ao'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating forex indicators: {str(e)}")
            return df

    def analyze_forex_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Analyze forex data and generate trading signals"""
        try:
            if len(df) < 50:
                return None
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev_2 = df.iloc[-3] if len(df) > 2 else prev
            
            # Initialize signal
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(self.ph_tz).isoformat(),
                'action': None,
                'entry_price': latest['close'],
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0,
                'reason': [],
                'timeframe': '1h',
                'risk_reward': self.risk_reward_ratio
            }
            
            confidence_score = 0
            reasons = []
            buy_signals = 0
            sell_signals = 0
            
            # RSI Analysis
            if latest['rsi'] < 30:
                buy_signals += 1
                confidence_score += 25
                reasons.append(f"RSI oversold at {latest['rsi']:.1f}")
            elif latest['rsi'] > 70:
                sell_signals += 1
                confidence_score += 25
                reasons.append(f"RSI overbought at {latest['rsi']:.1f}")
            elif 30 <= latest['rsi'] <= 35 and prev['rsi'] < 30:
                buy_signals += 1
                confidence_score += 15
                reasons.append(f"RSI recovering from oversold ({latest['rsi']:.1f})")
            elif 65 <= latest['rsi'] <= 70 and prev['rsi'] > 70:
                sell_signals += 1
                confidence_score += 15
                reasons.append(f"RSI declining from overbought ({latest['rsi']:.1f})")
            
            # MACD Analysis (very important for forex)
            if (latest['macd'] > latest['macd_signal'] and 
                prev['macd'] <= prev['macd_signal']):
                buy_signals += 1
                confidence_score += 30
                reasons.append("MACD bullish crossover")
            elif (latest['macd'] < latest['macd_signal'] and 
                  prev['macd'] >= prev['macd_signal']):
                sell_signals += 1
                confidence_score += 30
                reasons.append("MACD bearish crossover")
            
            # MACD histogram momentum
            if latest['macd_histogram'] > prev['macd_histogram'] > prev_2['macd_histogram']:
                buy_signals += 1
                confidence_score += 15
                reasons.append("MACD momentum strengthening (bullish)")
            elif latest['macd_histogram'] < prev['macd_histogram'] < prev_2['macd_histogram']:
                sell_signals += 1
                confidence_score += 15
                reasons.append("MACD momentum weakening (bearish)")
            
            # Stochastic Analysis
            if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
                buy_signals += 1
                confidence_score += 20
                reasons.append("Stochastic oversold")
            elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
                sell_signals += 1
                confidence_score += 20
                reasons.append("Stochastic overbought")
            
            # Stochastic crossover
            if (latest['stoch_k'] > latest['stoch_d'] and 
                prev['stoch_k'] <= prev['stoch_d'] and 
                latest['stoch_k'] < 50):
                buy_signals += 1
                confidence_score += 15
                reasons.append("Stochastic bullish crossover")
            elif (latest['stoch_k'] < latest['stoch_d'] and 
                  prev['stoch_k'] >= prev['stoch_d'] and 
                  latest['stoch_k'] > 50):
                sell_signals += 1
                confidence_score += 15
                reasons.append("Stochastic bearish crossover")
            
            # EMA Analysis (trend following for forex)
            if latest['close'] > latest['ema_8'] > latest['ema_21'] > latest['ema_50']:
                buy_signals += 1
                confidence_score += 25
                reasons.append("Strong bullish EMA alignment")
            elif latest['close'] < latest['ema_8'] < latest['ema_21'] < latest['ema_50']:
                sell_signals += 1
                confidence_score += 25
                reasons.append("Strong bearish EMA alignment")
            elif latest['close'] > latest['ema_21'] and prev['close'] <= prev['ema_21']:
                buy_signals += 1
                confidence_score += 20
                reasons.append("Price breaking above EMA21")
            elif latest['close'] < latest['ema_21'] and prev['close'] >= prev['ema_21']:
                sell_signals += 1
                confidence_score += 20
                reasons.append("Price breaking below EMA21")
            
            # Bollinger Bands Analysis
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            if bb_position <= 0.1:  # Near lower band
                buy_signals += 1
                confidence_score += 20
                reasons.append("Price near lower Bollinger Band")
            elif bb_position >= 0.9:  # Near upper band
                sell_signals += 1
                confidence_score += 20
                reasons.append("Price near upper Bollinger Band")
            
            # Bollinger Band squeeze detection
            if latest['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8:
                confidence_score += 10
                reasons.append("Bollinger Band squeeze - breakout potential")
            
            # Williams %R
            if latest['williams_r'] < -80:
                buy_signals += 1
                confidence_score += 15
                reasons.append("Williams %R oversold")
            elif latest['williams_r'] > -20:
                sell_signals += 1
                confidence_score += 15
                reasons.append("Williams %R overbought")
            
            # CCI Analysis
            if latest['cci'] < -100 and latest['cci'] > prev['cci']:
                buy_signals += 1
                confidence_score += 12
                reasons.append("CCI oversold reversal signal")
            elif latest['cci'] > 100 and latest['cci'] < prev['cci']:
                sell_signals += 1
                confidence_score += 12
                reasons.append("CCI overbought reversal signal")
            
            # Parabolic SAR (fixed pandas Series handling)
            try:
                if not pd.isna(latest['psar']) and not pd.isna(prev['psar']):
                    # Safely extract scalar values
                    current_psar = float(latest['psar'])
                    prev_psar = float(prev['psar'])
                    
                    if latest['close'] > current_psar and prev['close'] <= prev_psar:
                        buy_signals += 1
                        confidence_score += 18
                        reasons.append("Parabolic SAR bullish signal")
                    elif latest['close'] < current_psar and prev['close'] >= prev_psar:
                        sell_signals += 1
                        confidence_score += 18
                        reasons.append("Parabolic SAR bearish signal")
            except (ValueError, TypeError, KeyError):
                # Skip PSAR analysis if there are issues
                pass
            
            # 200 SMA trend filter (very important for forex)
            if latest['close'] > latest['sma_200']:
                if buy_signals > sell_signals:
                    confidence_score += 10
                    reasons.append("Above 200 SMA - bullish trend confirmation")
            elif latest['close'] < latest['sma_200']:
                if sell_signals > buy_signals:
                    confidence_score += 10
                    reasons.append("Below 200 SMA - bearish trend confirmation")
            
            # Trading session bonus (forex is time-sensitive)
            session_bonus = self.get_session_bonus()
            if session_bonus > 0:
                confidence_score += session_bonus
                reasons.append(f"Active trading session (+{session_bonus} confidence)")
            
            # Determine final action
            if buy_signals > sell_signals and buy_signals >= 2:
                signal['action'] = 'BUY'
            elif sell_signals > buy_signals and sell_signals >= 2:
                signal['action'] = 'SELL'
            else:
                signal['action'] = 'WAIT'
                confidence_score = max(15, confidence_score * 0.4)
                reasons.append("Mixed signals - recommend waiting")
            
            # Calculate Stop Loss and Take Profit using ATR (forex conservative approach)
            atr = latest['atr']
            volatility_mult = self.volatility_multipliers.get(symbol, 1.0)
            
            if signal['action'] == 'BUY':
                # ATR-based stop loss (more conservative for forex)
                sl_distance = atr * 1.5 * volatility_mult
                tp_distance = sl_distance * self.risk_reward_ratio
                
                signal['stop_loss'] = latest['close'] - sl_distance
                signal['take_profit'] = latest['close'] + tp_distance
                
            elif signal['action'] == 'SELL':
                sl_distance = atr * 1.5 * volatility_mult
                tp_distance = sl_distance * self.risk_reward_ratio
                
                signal['stop_loss'] = latest['close'] + sl_distance
                signal['take_profit'] = latest['close'] - tp_distance
            
            # Cap confidence at 100%
            signal['confidence'] = min(confidence_score, 100)
            signal['reason'] = reasons
            
            # Forex requires higher confidence threshold (50% minimum)
            if signal['action'] != 'WAIT' and signal['confidence'] >= 50:
                return signal
            elif signal['action'] == 'WAIT':
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing signal for {symbol}: {str(e)}")
            return None

    def get_session_bonus(self) -> int:
        """Get confidence bonus based on active trading sessions"""
        ph_now = datetime.now(self.ph_tz)
        current_hour = ph_now.hour
        
        # London session (15:00-24:00 PH time) - highest liquidity
        if 15 <= current_hour <= 23:
            return 15
        # London-NY overlap (21:00-24:00 PH time) - highest volatility
        elif 21 <= current_hour <= 23:
            return 20
        # NY session start (21:00-06:00 PH time)
        elif current_hour >= 21 or current_hour <= 6:
            return 10
        # Tokyo session (07:00-16:00 PH time) - good for JPY pairs
        elif 7 <= current_hour <= 16:
            return 8
        # Off-hours
        else:
            return 0

    async def get_forex_ai_analysis(self, signals: List[Dict]) -> str:
        """Get AI analysis specific to forex markets"""
        try:
            signals_summary = []
            for signal in signals:
                if signal['stop_loss'] and signal['take_profit']:
                    signals_summary.append(f"""
                    {signal['symbol']}: {signal['action']} at {signal['entry_price']:.5f}
                    Confidence: {signal['confidence']}%
                    SL: {signal['stop_loss']:.5f} | TP: {signal['take_profit']:.5f}
                    R:R Ratio: 1:{signal['risk_reward']:.1f}
                    Key factors: {', '.join(signal['reason'][:3])}
                    """)
                else:
                    signals_summary.append(f"""
                    {signal['symbol']}: {signal['action']} - {signal['confidence']}% confidence
                    Key factors: {', '.join(signal['reason'][:3])}
                    """)
            
            prompt = f"""
            As a professional forex trader, provide a brief market outlook for today's forex session.
            
            Current Time: {datetime.now(self.ph_tz).strftime('%Y-%m-%d %H:%M')} PH Time
            
            Today's Forex Signals (EUR/USD & USD/JPY focus):
            {''.join(signals_summary)}
            
            Provide a concise analysis (3-4 sentences) covering:
            1. Overall forex market sentiment and USD strength
            2. Key opportunities and risks for recommended trades
            3. Important economic events or session overlaps to watch
            
            Keep it professional and actionable for forex traders.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional forex trader. Provide concise, actionable forex market insights focusing on major currency pairs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting forex AI analysis: {str(e)}")
            return "AI forex analysis temporarily unavailable. Focus on technical signals and session timing."

    async def send_telegram_message(self, message: str, parse_mode: str = 'HTML'):
        """Send message to Telegram"""
        try:
            if not self.chat_id:
                await self.get_telegram_chat_id()
                
            if not self.chat_id:
                logger.warning("No chat ID available for Telegram")
                return False
                
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"Failed to send Telegram message: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    async def get_telegram_chat_id(self):
        """Get chat ID from Telegram updates"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['result']:
                            self.chat_id = data['result'][-1]['message']['chat']['id']
                            logger.info(f"Chat ID found: {self.chat_id}")
                        else:
                            logger.warning("No Telegram messages found. Send /start to the bot first.")
                    else:
                        logger.error(f"Failed to get Telegram updates: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error getting Telegram chat ID: {str(e)}")

    def format_forex_signals(self, signals: List[Dict], ai_analysis: str) -> str:
        """Format forex signals for Telegram"""
        
        ph_time = datetime.now(self.ph_tz)
        london_time = ph_time.astimezone(self.london_tz)
        ny_time = ph_time.astimezone(self.ny_tz)
        
        # Determine active session
        current_session = self.get_current_session()
        
        message_parts = [f"""📈 <b>FOREX TRADING SIGNALS</b> 💱
📅 <b>Date:</b> {ph_time.strftime('%Y-%m-%d')}
🕐 <b>PH Time:</b> {ph_time.strftime('%H:%M')}
🌍 <b>London:</b> {london_time.strftime('%H:%M')} | 🗽 <b>NY:</b> {ny_time.strftime('%H:%M')}
📊 <b>Active Session:</b> {current_session}
🎯 <b>Focus:</b> EUR/USD & USD/JPY

"""]
        
        for i, signal in enumerate(signals, 1):
            if signal['action'] == 'BUY':
                emoji = "🟢"
                action_emoji = "📈"
            elif signal['action'] == 'SELL':
                emoji = "🔴" 
                action_emoji = "📉"
            else:
                emoji = "⚪"
                action_emoji = "⏳"
            
            # Format forex prices (5 decimal places for most pairs)
            price_str = f"{signal['entry_price']:.5f}"
            
            if signal['stop_loss'] and signal['take_profit']:
                sl_str = f"{signal['stop_loss']:.5f}"
                tp_str = f"{signal['take_profit']:.5f}"
                
                # Calculate pip values
                if 'JPY' in signal['symbol']:
                    # For JPY pairs, 1 pip = 0.01
                    pip_multiplier = 100
                else:
                    # For other major pairs, 1 pip = 0.0001
                    pip_multiplier = 10000
                
                if signal['action'] == 'BUY':
                    sl_pips = (signal['entry_price'] - signal['stop_loss']) * pip_multiplier
                    tp_pips = (signal['take_profit'] - signal['entry_price']) * pip_multiplier
                else:
                    sl_pips = (signal['stop_loss'] - signal['entry_price']) * pip_multiplier
                    tp_pips = (signal['entry_price'] - signal['take_profit']) * pip_multiplier
                
                signal_text = f"""<b>{action_emoji} SIGNAL #{i}: {signal['symbol']}</b>
{emoji} <b>Action:</b> {signal['action']}
💰 <b>Entry:</b> {price_str}
🛑 <b>Stop Loss:</b> {sl_str} ({sl_pips:.1f} pips)
🎯 <b>Take Profit:</b> {tp_str} ({tp_pips:.1f} pips)
📊 <b>Confidence:</b> {signal['confidence']}%
⚖️ <b>Risk:Reward:</b> 1:{signal['risk_reward']:.1f}

"""
            else:
                signal_text = f"""<b>{action_emoji} SIGNAL #{i}: {signal['symbol']}</b>
{emoji} <b>Action:</b> {signal['action']}
💰 <b>Price:</b> {price_str}
📊 <b>Confidence:</b> {signal['confidence']}%

"""
            
            # Add technical reasons
            if signal['reason']:
                signal_text += f"""<b>Technical Setup:</b>
{chr(10).join(f"• {reason}" for reason in signal['reason'][:4])}

"""
            
            message_parts.append(signal_text)
        
        message_parts.append(f"""<b>🤖 AI Forex Market Analysis:</b>
{ai_analysis}

<b>⚠️ Risk Management:</b>
• Max 2% risk per trade
• Always use stop losses  
• Monitor economic news
• Respect session timing

<b>📊 Trading Sessions (PH Time):</b>
• 🗾 Tokyo: 07:00-16:00
• 🌍 London: 15:00-24:00 (High Liquidity)
• 🗽 NY: 21:00-06:00
• 🔥 Overlap: 21:00-24:00 (Best Volatility)

<i>⚠️ Forex carries significant risk. This is not financial advice. Trade responsibly.</i>""")
        
        return ''.join(message_parts)

    def get_current_session(self) -> str:
        """Get current active trading session"""
        ph_now = datetime.now(self.ph_tz)
        hour = ph_now.hour
        
        if 21 <= hour <= 23:
            return "London-NY Overlap 🔥"
        elif 15 <= hour <= 23:
            return "London Session 🌍"
        elif hour >= 21 or hour <= 6:
            return "New York Session 🗽"
        elif 7 <= hour <= 16:
            return "Tokyo Session 🗾"
        else:
            return "Low Activity Period 😴"

    async def generate_forex_signals(self):
        """Generate forex trading signals"""
        logger.info("Generating forex trading signals...")
        
        signals = []
        
        for symbol in self.symbols:
            try:
                # Get 1-hour data for forex analysis
                df = await self.get_forex_data(symbol, interval='1h', outputsize=100)
                
                if df is None or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Calculate forex-specific indicators
                df = self.calculate_forex_indicators(df)
                
                # Analyze for signals
                signal = self.analyze_forex_signal(df, symbol)
                
                if signal and signal['action'] != 'WAIT':
                    signals.append(signal)
                    logger.info(f"Forex signal generated for {symbol}: {signal['action']} (Confidence: {signal['confidence']}%)")
                
                # Delay between API calls
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue
        
        if signals:
            # Get AI forex analysis
            ai_analysis = await self.get_forex_ai_analysis(signals)
            
            # Format and send message
            message = self.format_forex_signals(signals, ai_analysis)
            await self.send_telegram_message(message)
            
            logger.info(f"Forex signals sent for {len(signals)} pairs")
        else:
            # Send no signals message
            current_session = self.get_current_session()
            no_signal_msg = f"""📈 <b>FOREX MARKET UPDATE</b>

📅 <b>Date:</b> {datetime.now(self.ph_tz).strftime('%Y-%m-%d %H:%M')} PH Time
💱 <b>Pairs Analyzed:</b> EUR/USD, USD/JPY
📊 <b>Session:</b> {current_session}

⚪ <b>Status:</b> No high-confidence trading setups detected
📉 <b>Market Condition:</b> Consolidation/Mixed signals on major pairs

<b>Recommendation:</b> Wait for clearer breakouts or session openings. Monitor economic news and central bank communications.

<i>📈 Best forex opportunities often come during session overlaps. Stay patient for quality setups.</i>"""
            
            await self.send_telegram_message(no_signal_msg)
            logger.info("No high-confidence forex signals - sent market update")

    async def send_startup_message(self):
        """Send forex agent startup message"""
        startup_message = f"""📈 <b>FOREX TRADING AGENT ACTIVATED</b>

📅 <b>Started:</b> {datetime.now(self.ph_tz).strftime('%Y-%m-%d %H:%M')} PH Time
💱 <b>Monitoring:</b> EUR/USD, USD/JPY
⚡ <b>Frequency:</b> Every 3 hours + Session alerts
📊 <b>Timeframes:</b> 1H (main analysis)
🎯 <b>Strategy:</b> Major pair momentum & trend following

<b>🔥 Trading Schedule:</b>
• <b>Tokyo Session:</b> 07:00-16:00 PH (JPY focus)
• <b>London Session:</b> 15:00-24:00 PH (EUR focus) 
• <b>NY Session:</b> 21:00-06:00 PH (USD strength)
• <b>Best Opportunities:</b> 21:00-24:00 PH (London-NY overlap)

<i>Ready for professional forex trading! 💪</i>"""
        
        await self.send_telegram_message(startup_message)

    def run_async_task(self, coro):
        """Helper function to run async tasks from synchronous scheduler"""
        if self.main_loop and not self.main_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
            try:
                future.result(timeout=300)  # 5 minute timeout
            except Exception as e:
                logger.error(f"Error running scheduled task: {str(e)}")
        else:
            logger.error("Main event loop is not available for scheduled tasks")

    def schedule_forex_signals(self):
        """Schedule forex signal generation based on trading sessions"""
        # Generate signals every 3 hours during active sessions
        for hour in [9, 12, 15, 18, 21, 0]:  # PH time (0 = midnight, not 24)
            schedule.every().day.at(f"{hour:02d}:00").do(lambda: self.run_async_task(self.generate_forex_signals()))
        
        # Session opening alerts
        schedule.every().day.at("07:00").do(lambda: self.run_async_task(self.send_session_alert("Tokyo")))
        schedule.every().day.at("15:00").do(lambda: self.run_async_task(self.send_session_alert("London")))
        schedule.every().day.at("21:00").do(lambda: self.run_async_task(self.send_session_alert("New York")))
        
        # Weekly forex market overview
        schedule.every().sunday.at("10:00").do(lambda: self.run_async_task(self.send_weekly_forex_overview()))
        
        logger.info("Forex signals scheduled: Every 3 hours + Session alerts")

    async def send_session_alert(self, session_name: str):
        """Send trading session opening alert"""
        ph_time = datetime.now(self.ph_tz)
        
        session_info = {
            'Tokyo': {
                'emoji': '🗾',
                'focus': 'USD/JPY opportunities',
                'characteristics': 'Lower volatility, range trading'
            },
            'London': {
                'emoji': '🌍', 
                'focus': 'EUR/USD momentum',
                'characteristics': 'High liquidity, trend continuation'
            },
            'New York': {
                'emoji': '🗽',
                'focus': 'USD strength assessment', 
                'characteristics': 'Highest volatility with London overlap'
            }
        }
        
        info = session_info.get(session_name, {})
        
        alert_msg = f"""{info.get('emoji', '📊')} <b>{session_name.upper()} SESSION OPENING</b>

🕐 <b>Time:</b> {ph_time.strftime('%H:%M')} PH
🎯 <b>Focus:</b> {info.get('focus', 'Market opportunities')}
📈 <b>Characteristics:</b> {info.get('characteristics', 'Active trading period')}

<b>Action Items:</b>
• Review technical setups
• Monitor economic news
• Prepare for increased volatility
• Check position sizes

<i>Session trading activated! 🚀</i>"""
        
        await self.send_telegram_message(alert_msg)

    async def send_weekly_forex_overview(self):
        """Send weekly forex market overview"""
        weekly_msg = f"""📊 <b>WEEKLY FOREX MARKET OVERVIEW</b>

📅 <b>Week of:</b> {datetime.now(self.ph_tz).strftime('%Y-%m-%d')}
💱 <b>Focus:</b> EUR/USD & USD/JPY analysis

<b>📈 This Week's Forex Strategy:</b>
• Monitor central bank communications
• Watch for key support/resistance levels
• Follow USD strength/weakness themes
• Session-based trading approach

<b>🗓️ Key Economic Events:</b>
• Check economic calendar for high-impact news
• Fed, ECB, BOJ policy updates
• Employment, inflation, GDP data releases

<b>⚠️ Risk Reminders:</b>
• Forex leverage amplifies both gains and losses
• Use appropriate position sizing (2% max risk)
• Set stop losses on all trades
• Stay informed on geopolitical events

<i>📈 Trade with discipline and follow your plan!</i>"""
        
        await self.send_telegram_message(weekly_msg)

    def run_scheduler(self):
        """Run scheduled jobs in a separate thread"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    async def start(self):
        """Start the forex trading agent"""
        logger.info("Starting Forex Trading Agent...")
        
        # Store the main event loop reference
        self.main_loop = asyncio.get_running_loop()
        
        # Get initial chat ID
        await self.get_telegram_chat_id()
        
        # Send startup message
        await self.send_startup_message()
        
        # Schedule trading jobs
        self.schedule_forex_signals()
        
        # Start scheduler in background thread
        scheduler_thread = Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Generate initial signals immediately when starting
        logger.info("Generating initial forex signals...")
        await self.generate_forex_signals()
        
        # Keep the main thread alive
        try:
            while True:
                await asyncio.sleep(60)  # Sleep for 1 minute
        except KeyboardInterrupt:
            logger.info("Forex trading agent stopped by user")

def main():
    """Main function to run the forex trading agent"""
    agent = ForexTradingAgent()
    
    try:
        # Run the agent
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nForex trading agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()