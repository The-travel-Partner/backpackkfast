"""
Cloudflare Images CDN integration for uploading and serving images.
"""
import aiohttp
import hashlib
import os
from typing import Optional

# Cloudflare Images API Configuration
# You can use EITHER API Token OR Global API Key (not both)

# Option 1: API Token (recommended) - requires "Cloudflare Images:Edit" permission
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "dfa0c3871abf9933dbd76a448a18b8f5")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

# Option 2: Global API Key - requires your account email as well
CLOUDFLARE_GLOBAL_API_KEY = os.getenv("CLOUDFLARE_GLOBAL_API_KEY", "f35561ab539311231607c7e61a6e675633697")
CLOUDFLARE_EMAIL = os.getenv("CLOUDFLARE_EMAIL", "contact@backpackk.com")

CLOUDFLARE_IMAGES_DELIVERY_URL = f"https://imagedelivery.net/{CLOUDFLARE_ACCOUNT_ID}"

# Base URL for Cloudflare Images API
CLOUDFLARE_IMAGES_API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/images/v1"


def get_auth_headers() -> dict:
    """
    Get authentication headers based on available credentials.
    Supports both API Token and Global API Key methods.
    """
    if CLOUDFLARE_API_TOKEN:
        # Use API Token (Bearer token)
        return {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
    elif CLOUDFLARE_GLOBAL_API_KEY and CLOUDFLARE_EMAIL:
        # Use Global API Key (requires email + key)
        return {
            "X-Auth-Email": CLOUDFLARE_EMAIL,
            "X-Auth-Key": CLOUDFLARE_GLOBAL_API_KEY
        }
    else:
        raise ValueError(
            "No Cloudflare credentials configured. Set either:\n"
            "  - CLOUDFLARE_API_TOKEN (recommended), or\n"
            "  - CLOUDFLARE_GLOBAL_API_KEY + CLOUDFLARE_EMAIL"
        )


async def upload_image_to_cloudflare(
    image_bytes: bytes,
    filename: Optional[str] = None,
    metadata: Optional[dict] = None
) -> Optional[dict]:
    """
    Upload an image to Cloudflare Images.
    
    Args:
        image_bytes: The raw image bytes to upload
        filename: Optional filename for the image
        metadata: Optional metadata to attach to the image
        
    Returns:
        dict with 'id' and 'url' keys if successful, None otherwise
    """
    if not filename:
        # Generate a unique filename based on image content hash
        image_hash = hashlib.md5(image_bytes).hexdigest()
        filename = f"photo_{image_hash}.jpg"
    
    headers = get_auth_headers()
    
    # Prepare multipart form data
    form_data = aiohttp.FormData()
    form_data.add_field(
        'file',
        image_bytes,
        filename=filename,
        content_type='image/jpeg'
    )
    
    if metadata:
        import json
        form_data.add_field('metadata', json.dumps(metadata))
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CLOUDFLARE_IMAGES_API_URL,
                headers=headers,
                data=form_data
            ) as response:
                result = await response.json()
                
                if response.status == 200 and result.get('success'):
                    image_data = result.get('result', {})
                    image_id = image_data.get('id')
                    
                    # Construct the delivery URL
                    # The 'public' variant is commonly used, adjust as needed
                    cdn_url = f"{CLOUDFLARE_IMAGES_DELIVERY_URL}/{image_id}/public"
                    
                    return {
                        'id': image_id,
                        'url': cdn_url,
                        'variants': image_data.get('variants', [])
                    }
                else:
                    print(f"❌ Cloudflare upload failed: {result.get('errors', 'Unknown error')}")
                    return None
                    
    except Exception as e:
        print(f"❌ Error uploading to Cloudflare: {str(e)}")
        return None


async def upload_images_batch(
    images: list[bytes],
    place_id: Optional[str] = None
) -> list[dict]:
    """
    Upload multiple images to Cloudflare Images.
    
    Args:
        images: List of image bytes to upload
        place_id: Optional place_id to include in metadata
        
    Returns:
        List of dicts with 'id' and 'url' keys for successful uploads
    """
    results = []
    
    for i, image_bytes in enumerate(images):
        metadata = {}
        if place_id:
            metadata['place_id'] = place_id
            metadata['index'] = i
        
        result = await upload_image_to_cloudflare(
            image_bytes=image_bytes,
            filename=f"{place_id}_{i}.jpg" if place_id else None,
            metadata=metadata if metadata else None
        )
        
        if result:
            results.append(result)
            print(f"✅ Uploaded image {i+1}/{len(images)} to Cloudflare CDN")
        else:
            print(f"⚠️ Failed to upload image {i+1}/{len(images)}")
    
    return results


async def delete_image_from_cloudflare(image_id: str) -> bool:
    """
    Delete an image from Cloudflare Images.
    
    Args:
        image_id: The Cloudflare image ID to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    headers = get_auth_headers()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{CLOUDFLARE_IMAGES_API_URL}/{image_id}",
                headers=headers
            ) as response:
                result = await response.json()
                return response.status == 200 and result.get('success', False)
    except Exception as e:
        print(f"❌ Error deleting from Cloudflare: {str(e)}")
        return False
