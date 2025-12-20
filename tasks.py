"""
Celery Tasks for Backpackk FastAPI
Contains background tasks for image processing and uploading.
"""
import asyncio
from datetime import datetime
from celery_app import celery_app
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB connection (same as config.py)
MONGODB_URL = "mongodb+srv://nimishspslosal:nimish123@cluster0.0c1ay.mongodb.net/?retryWrites=true&w=majority"


def get_mongo_client():
    """Get a new MongoDB client for the task."""
    return AsyncIOMotorClient(MONGODB_URL)


async def _upsert_images_async(place_id: str, photo_references: list) -> dict:
    """
    Async implementation of image upsert logic.
    Fetches photos from Google Places API, uploads to Cloudflare CDN,
    and upserts the CDN URLs with place_id to MongoDB images collection.
    """
    from tripgen.asyncclass import RetrievePhotos
    from cloudflare_cdn import upload_images_batch
    
    print(f"📷 [Celery Task] Processing {len(photo_references)} photos for place_id: {place_id}")
    
    # Fetch photos using RetrievePhotos
    photos = RetrievePhotos(photo_references, place_id)
    images = await photos.get_photos()
    
    # Collect raw image bytes for Cloudflare upload
    raw_images = []
    for response in images:
        image_bytes = b''
        
        if hasattr(response, 'body'):
            image_bytes = response.body
        elif hasattr(response, 'content'):
            image_bytes = response.content
        
        if image_bytes:
            raw_images.append(image_bytes)
    
    if not raw_images:
        return {"success": False, "message": "No images found to upload", "cdn_urls": []}
    
    # Upload images to Cloudflare CDN with place_id
    print(f"📤 [Celery Task] Uploading {len(raw_images)} images to Cloudflare CDN...")
    cdn_results = await upload_images_batch(raw_images, place_id)
    print(f"✅ [Celery Task] Successfully uploaded {len(cdn_results)} images to CDN")
    
    # Extract CDN URLs
    cdn_urls = [result["url"] for result in cdn_results]
    
    # Upsert to MongoDB images collection
    client = get_mongo_client()
    images_collection = client.backpackk.images
    
    # Create the document to upsert
    image_doc = {
        "place_id": place_id,
        "cdn_urls": cdn_urls,
        "cdn_results": cdn_results,  # Store full results including IDs
        "timestamp": datetime.now()
    }
    
    # Upsert: update if exists, insert if not
    result = await images_collection.update_one(
        {"place_id": place_id},
        {"$set": image_doc},
        upsert=True
    )
    
    upsert_type = "inserted" if result.upserted_id else "updated"
    print(f"✅ [Celery Task] Successfully {upsert_type} document for place_id: {place_id}")
    
    # Close the client
    client.close()
    
    return {
        "success": True,
        "place_id": place_id,
        "cdn_urls": cdn_urls,
        "images_uploaded": len(cdn_urls),
        "upsert_type": upsert_type
    }


@celery_app.task(bind=True, name="tasks.upsert_images_task")
def upsert_images_task(self, place_id: str, photo_references: list) -> dict:
    """
    Celery task to upsert images for a place.
    
    Args:
        place_id: The Google Places place_id
        photo_references: List of photo reference strings from Google Places API
        
    Returns:
        dict with success status, cdn_urls, and upsert details
    """
    try:
        # Run the async function in a new event loop
        result = asyncio.run(_upsert_images_async(place_id, photo_references))
        return result
    except Exception as e:
        print(f"❌ [Celery Task] Error processing images for {place_id}: {str(e)}")
        # Retry on failure (max 3 retries with exponential backoff)
        raise self.retry(exc=e, countdown=2 ** self.request.retries, max_retries=3)
