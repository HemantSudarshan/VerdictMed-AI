"""
Encryption Service
Handle transparent encryption/decryption of PII data using Fernet (symmetric encryption).
"""

from cryptography.fernet import Fernet
from typing import Optional
from loguru import logger
from src.config import get_settings


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryption service.
        
        Args:
            key: Fernet key (32 url-safe base64-encoded bytes). 
                 If None, loads from config.
        """
        settings = get_settings()
        self.key = key or settings.encryption_key
        
        try:
            # Ensure key is bytes
            if isinstance(self.key, str):
                self.key = self.key.encode()
                
            self.cipher_suite = Fernet(self.key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption service: {e}")
            # Generate a temporary key to prevent crash, but warn loudly
            logger.critical("USING TEMPORARY ENCRYPTION KEY - DATA WILL BE UNREADABLE AFTER RESTART")
            self.key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.key)
            
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not data:
            return ""
        try:
            encrypted_bytes = self.cipher_suite.encrypt(data.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError("Encryption failed")
            
    def decrypt(self, token: str) -> str:
        """Decrypt string token"""
        if not token:
            return ""
        try:
            decrypted_bytes = self.cipher_suite.decrypt(token.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Decryption failed - invalid key or corrupted data")
            
    @staticmethod
    def generate_key() -> str:
        """Generate a new valid key"""
        return Fernet.generate_key().decode()


# Singleton instance
_encryption_service = None

def get_encryption_service() -> EncryptionService:
    """Get singleton encryption service instance"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service
