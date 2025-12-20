"""
Standalone Cron Job for Google Cloud
This is a dummy cron job that can be deployed to Google Cloud Run
and triggered by Cloud Scheduler.
"""

import os
import logging
from datetime import datetime
from flask import Flask, request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def run_cron_job():
    """
    Main cron job endpoint.
    Cloud Scheduler will hit this endpoint to trigger the job.
    """
    logger.info("Cron job triggered at %s", datetime.utcnow().isoformat())
    
    try:
        # ============================================
        # YOUR CRON JOB LOGIC GOES HERE
        # ============================================
        
        # Dummy task - replace with actual logic
        result = perform_dummy_task()
        
        logger.info("Cron job completed successfully: %s", result)
        return {"status": "success", "message": result, "timestamp": datetime.utcnow().isoformat()}, 200
    
    except Exception as e:
        logger.error("Cron job failed: %s", str(e))
        return {"status": "error", "message": str(e), "timestamp": datetime.utcnow().isoformat()}, 500


def perform_dummy_task():
    """
    Dummy task placeholder.
    Replace this with your actual cron job logic.
    """
    logger.info("Performing dummy task...")
    
    # Example tasks you might do:
    # - Sync data between databases
    # - Send scheduled notifications
    # - Clean up old records
    # - Generate reports
    # - Call external APIs
    
    return "Dummy task executed successfully"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}, 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
