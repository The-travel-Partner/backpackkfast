# Google Cloud Cron Job

A standalone Python cron job designed for deployment on Google Cloud Run with Cloud Scheduler.

## Structure

```
cron/
├── main.py           # Flask app with cron job logic
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container configuration
├── cloudbuild.yaml   # Cloud Build CI/CD config
└── README.md         # This file
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py

# Test the endpoint
curl http://localhost:8080/
curl http://localhost:8080/health
```

## Deployment to Google Cloud

### 1. Deploy to Cloud Run

```bash
# Navigate to cron directory
cd cron

# Deploy using gcloud
gcloud run deploy cron-job \
  --source . \
  --region us-central1 \
  --no-allow-unauthenticated \
  --memory 256Mi \
  --timeout 300s
```

### 2. Create Cloud Scheduler Job

```bash
# Create a scheduler job that runs every hour
gcloud scheduler jobs create http cron-job-trigger \
  --location us-central1 \
  --schedule "0 * * * *" \
  --uri "YOUR_CLOUD_RUN_URL" \
  --http-method POST \
  --oidc-service-account-email YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com
```

### Common Cron Schedules

| Schedule | Cron Expression |
|----------|-----------------|
| Every minute | `* * * * *` |
| Every hour | `0 * * * *` |
| Every day at midnight | `0 0 * * *` |
| Every Monday at 9am | `0 9 * * 1` |
| Every 1st of month | `0 0 1 * *` |

## Environment Variables

Set these in Cloud Run if needed:

- `PORT` - Server port (default: 8080)
- Add your custom env vars as needed

## Adding Your Logic

Edit the `perform_dummy_task()` function in `main.py` to add your actual cron job logic.
