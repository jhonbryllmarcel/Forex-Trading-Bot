from flask import Flask
from threading import Thread
from forex_trading_agent import ForexTradingAgent
import asyncio
import os  # <- added to read environment variable

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
    # Use Render's dynamic port, fallback to 10000 for local testing
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
