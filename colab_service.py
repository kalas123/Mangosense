import os
import requests
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
import io
import base64
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ColabService:
    """Service to communicate with Google Colab inference server"""
    
    def __init__(self, colab_url: Optional[str] = None, timeout: int = 120, max_retries: int = 3):
        """
        Initialize Colab service
        
        Args:
            colab_url: URL of the Colab inference server (from ngrok)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.colab_url = colab_url or os.getenv('COLAB_API_URL')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.last_health_check = None
        self.is_healthy = False
        
        if not self.colab_url:
            logger.warning("No Colab API URL provided. Set COLAB_API_URL environment variable.")
    
    def is_available(self) -> bool:
        """Check if Colab service is available"""
        if not self.colab_url:
            return False
        
        # Use cached health check if recent (within 1 minute)
        if (self.last_health_check and 
            datetime.now() - self.last_health_check < timedelta(minutes=1) and 
            self.is_healthy):
            return True
        
        return self.health_check()
    
    def health_check(self) -> bool:
        """Perform health check on Colab service"""
        if not self.colab_url:
            return False
        
        try:
            response = self.session.get(
                f"{self.colab_url}/health",
                timeout=10  # Short timeout for health check
            )
            
            if response.status_code == 200:
                health_data = response.json()
                self.is_healthy = health_data.get('status') == 'healthy'
                self.last_health_check = datetime.now()
                
                logger.info(f"Colab service health: {health_data}")
                return self.is_healthy
            else:
                logger.warning(f"Colab health check failed: {response.status_code}")
                self.is_healthy = False
                return False
                
        except Exception as e:
            logger.error(f"Colab health check error: {str(e)}")
            self.is_healthy = False
            return False
    
    def process_image(self, image_path: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Process image through Colab service
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (result_dict, error_message)
        """
        if not self.is_available():
            return None, "Colab service is not available"
        
        try:
            # Prepare image for upload
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
                
                # Make request with retries
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"Processing image via Colab (attempt {attempt + 1})")
                        
                        response = self.session.post(
                            f"{self.colab_url}/process_image",
                            files=files,
                            timeout=self.timeout
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info("Image processed successfully via Colab")
                            return result, None
                        else:
                            error_msg = f"Colab processing failed: {response.status_code}"
                            if response.content:
                                try:
                                    error_data = response.json()
                                    error_msg += f" - {error_data.get('error', 'Unknown error')}"
                                except:
                                    error_msg += f" - {response.text[:200]}"
                            
                            logger.error(error_msg)
                            
                            # Don't retry on client errors (4xx)
                            if 400 <= response.status_code < 500:
                                return None, error_msg
                            
                            # Retry on server errors (5xx) and timeouts
                            if attempt == self.max_retries - 1:
                                return None, error_msg
                            
                    except requests.exceptions.Timeout:
                        logger.warning(f"Colab request timeout (attempt {attempt + 1})")
                        if attempt == self.max_retries - 1:
                            return None, "Colab service timeout"
                    
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Colab request error (attempt {attempt + 1}): {str(e)}")
                        if attempt == self.max_retries - 1:
                            return None, f"Colab service error: {str(e)}"
                    
                    # Reset file pointer for retry
                    f.seek(0)
                    
        except Exception as e:
            logger.error(f"Error preparing image for Colab: {str(e)}")
            return None, f"Image preparation error: {str(e)}"
        
        return None, "Unexpected error in Colab processing"
    
    def batch_process_images(self, image_paths: List[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Process multiple images through Colab service
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Tuple of (result_dict, error_message)
        """
        if not self.is_available():
            return None, "Colab service is not available"
        
        try:
            # Prepare images for upload
            files = []
            for image_path in image_paths:
                with open(image_path, 'rb') as f:
                    files.append(('images', (os.path.basename(image_path), f.read(), 'image/jpeg')))
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Batch processing {len(image_paths)} images via Colab (attempt {attempt + 1})")
                    
                    response = self.session.post(
                        f"{self.colab_url}/batch_process",
                        files=files,
                        timeout=self.timeout * 2  # Double timeout for batch processing
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"Batch processed {result.get('total_processed', 0)} images via Colab")
                        return result, None
                    else:
                        error_msg = f"Colab batch processing failed: {response.status_code}"
                        if response.content:
                            try:
                                error_data = response.json()
                                error_msg += f" - {error_data.get('error', 'Unknown error')}"
                            except:
                                error_msg += f" - {response.text[:200]}"
                        
                        logger.error(error_msg)
                        
                        # Don't retry on client errors (4xx)
                        if 400 <= response.status_code < 500:
                            return None, error_msg
                        
                        # Retry on server errors (5xx) and timeouts
                        if attempt == self.max_retries - 1:
                            return None, error_msg
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Colab batch request timeout (attempt {attempt + 1})")
                    if attempt == self.max_retries - 1:
                        return None, "Colab service timeout"
                
                except requests.exceptions.RequestException as e:
                    logger.error(f"Colab batch request error (attempt {attempt + 1}): {str(e)}")
                    if attempt == self.max_retries - 1:
                        return None, f"Colab service error: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error preparing images for Colab batch processing: {str(e)}")
            return None, f"Batch preparation error: {str(e)}"
        
        return None, "Unexpected error in Colab batch processing"
    
    def update_url(self, new_url: str):
        """Update the Colab service URL"""
        self.colab_url = new_url
        self.last_health_check = None
        self.is_healthy = False
        logger.info(f"Updated Colab URL to: {new_url}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the Colab service"""
        return {
            'url': self.colab_url,
            'is_available': self.is_available(),
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }

# Global instance
colab_service = ColabService()

def get_colab_service() -> ColabService:
    """Get the global Colab service instance"""
    return colab_service