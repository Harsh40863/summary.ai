import redis
import requests
import hashlib
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RedisTranslationService:
    """Fast translation service with Redis caching"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, cache_ttl: int = 86400):
        """Initialize Redis translation service"""
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=redis_db, 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        self.cache_ttl = cache_ttl
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese'
        }
        
        # MyMemory API - FREE translation service
        self.api_url = "https://api.mymemory.translated.net/get"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("✅ Redis translation service connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, translations will be slower: {e}")
    
    def _generate_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"trans:{source_lang}:{target_lang}:{text_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get translation from cache"""
        try:
            return self.redis_client.get(cache_key)
        except:
            return None
    
    def _save_to_cache(self, cache_key: str, translation: str):
        """Save translation to cache"""
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, translation)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
    
    def _call_api(self, text: str, source_lang: str, target_lang: str) -> str:
        """Call MyMemory translation API (FREE)"""
        try:
            params = {'q': text, 'langpair': f"{source_lang}|{target_lang}"}
            response = requests.get(self.api_url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                return data['responseData']['translatedText']
            else:
                return text
                
        except Exception as e:
            logger.debug(f"Translation API failed: {e}")
            return text
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = 'en') -> str:
        """Main translation function with caching"""
        # Skip if same language or invalid
        if (not text or 
            source_lang == target_lang or 
            target_lang not in self.supported_languages or
            len(text.strip()) < 2):
            return text
        
        # Generate cache key
        cache_key = self._generate_cache_key(text, source_lang, target_lang)
        
        # Try cache first (fastest - 1-2ms)
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
        
        # Call API (slower - 100-300ms)
        translation = self._call_api(text, source_lang, target_lang)
        
        # Cache result
        if translation != text:
            self._save_to_cache(cache_key, translation)
        
        return translation
    
    def translate_response(self, response_data: Dict[str, Any], 
                          target_lang: str, source_lang: str = 'en') -> Dict[str, Any]:
        """Translate all text fields in API response"""
        if target_lang == source_lang or target_lang not in self.supported_languages:
            return response_data
        
        # Copy response
        translated = response_data.copy()
        
        # Translate main message
        if 'message' in translated:
            translated['message'] = self.translate_text(
                translated['message'], target_lang, source_lang
            )
        
        # Translate results array
        if 'results' in translated and isinstance(translated['results'], list):
            translated_results = []
            
            for result in translated['results']:
                if isinstance(result, dict):
                    new_result = result.copy()
                    
                    # Translate common fields
                    fields_to_translate = [
                        'best_sentence', 'summary', 'refined_insight',
                        'description', 'content', 'answer'
                    ]
                    
                    for field in fields_to_translate:
                        if field in new_result and isinstance(new_result[field], str):
                            new_result[field] = self.translate_text(
                                new_result[field], target_lang, source_lang
                            )
                    
                    translated_results.append(new_result)
                else:
                    translated_results.append(result)
            
            translated['results'] = translated_results
        
        # Add translation metadata
        translated['translation'] = {
            'target_language': target_lang,
            'target_language_name': self.supported_languages.get(target_lang, target_lang),
            'source_language': source_lang,
            'translated': True
        }
        
        return translated
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages"""
        return self.supported_languages.copy()
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages

# Global instance and helper functions
_translator_instance = None

def get_translator() -> RedisTranslationService:
    """Get or create translator instance"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = RedisTranslationService(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6379)),
            cache_ttl=int(os.getenv('TRANSLATION_CACHE_TTL', 86400))
        )
    return _translator_instance

def translate_text(text: str, target_lang: str, source_lang: str = 'en') -> str:
    """Simple function to translate text"""
    translator = get_translator()
    return translator.translate_text(text, target_lang, source_lang)

def translate_response(response_data: Dict[str, Any], target_lang: str, source_lang: str = 'en') -> Dict[str, Any]:
    """Simple function to translate API response"""
    translator = get_translator()
    return translator.translate_response(response_data, target_lang, source_lang)

def get_supported_languages() -> Dict[str, str]:
    """Get supported languages"""
    translator = get_translator()
    return translator.get_supported_languages()

def is_supported_language(lang_code: str) -> bool:
    """Check if language is supported"""
    translator = get_translator()
    return translator.is_supported_language(lang_code)