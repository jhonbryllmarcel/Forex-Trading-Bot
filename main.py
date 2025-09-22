from flask import Flask, jsonify
from threading import Thread
from forex_trading_agent import ForexTradingAgent
import asyncio
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to track agent status
agent_status = {"running": False, "last_signal": None, "error": None}
agent = None

def run_bot():
    """Run the trading bot in background"""
    global agent_status, agent
    try:
        agent = ForexTradingAgent()
        agent_status["running"] = True
        agent_status["error"] = None
        logger.info("Starting Forex Trading Agent...")
        asyncio.run(agent.start())
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        agent_status["running"] = False
        agent_status["error"] = str(e)

# Start the bot in a background thread
bot_thread = Thread(target=run_bot, daemon=True)
bot_thread.start()

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "Forex Trading Agent Web Service",
        "bot_running": agent_status["running"],
        "last_error": agent_status.get("error"),
        "message": "Bot is running in background and will send Telegram alerts"
    })

@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        "service": "healthy",
        "bot_status": agent_status,
        "bot_thread_alive": bot_thread.is_alive() if 'bot_thread' in globals() else False
    })

@app.route('/ping')
def ping():
    """Simple ping endpoint for external monitoring (UptimeRobot)"""
    return jsonify({
        "status": "alive",
        "timestamp": time.time(),
        "bot_running": agent_status["running"],
        "uptime": "ok"
    })

@app.route('/restart')
def restart_bot():
    """Endpoint to restart the bot if needed"""
    global bot_thread, agent_status
    try:
        if bot_thread.is_alive():
            logger.info("Bot thread is still running")
            return jsonify({"status": "Bot is already running"})
        else:
            logger.info("Restarting bot thread...")
            bot_thread = Thread(target=run_bot, daemon=True)
            bot_thread.start()
            return jsonify({"status": "Bot restarted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    # Use Render's dynamic port, fallback to 10000 for local testing
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Flask app on port {port}")
    logger.info("Ready for external monitoring (UptimeRobot recommended)")
    app.run(host="0.0.0.0", port=port, debug=False)