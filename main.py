from flask import Flask
from threading import Thread
from forex_trading_agent import ForexTradingAgent
import asyncio

app = Flask(__name__)

agent = ForexTradingAgent()

def run_bot():
    asyncio.run(agent.start())

# Run the bot in a background thread
Thread(target=run_bot, daemon=True).start()

@app.route('/')
def index():
    return "Forex Trading Agent is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
