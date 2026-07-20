import os
import json
import time
import hashlib
import google.generativeai as genai

# Sadə in-memory cache
# Format: {"hash_key": (data_dict, timestamp)}
_cache = {}
CACHE_TTL = 900  # 15 dəqiqə (saniyə ilə)

def _get_cache_key(facts: dict) -> str:
    # Açar-dəyər ardıcıllığının həmişə eyni olması üçün sort_keys=True
    facts_str = json.dumps(facts, sort_keys=True)
    return hashlib.md5(facts_str.encode('utf-8')).hexdigest()

def _get_from_cache(key: str):
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        else:
            del _cache[key]
    return None

def _set_to_cache(key: str, data: dict):
    _cache[key] = (data, time.time())

def generate_narrative(facts: dict) -> dict:
    default_model = "gemini-2.5-flash"
    model_name = os.environ.get("GEMINI_MODEL", default_model)
    
    default_response = {
        "text": None,
        "generatedBy": None,
        "disclaimer": None,
        "success": False
    }
    
    try:
        cache_key = _get_cache_key(facts)
        cached_data = _get_from_cache(cache_key)
        if cached_data:
            return cached_data

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[Narrative] xəta — səbəb: GEMINI_API_KEY tapılmadı")
            return default_response
            
        genai.configure(api_key=api_key)
        
        system_instruction = """Sən avtomobil telematikası və təhlükəsizlik sistemləri üçün hesabat hazırlayan neytral köməkçisən.
Aşağıdakı qaydalara CİDDİ şəkildə əməl et:
1. Yalnız və yalnız sənə verilən faktlara əsaslan, heç nə uydurma.
2. Naməlum və ya "None" olan faktları sadəcə qeyd etmə, onlar barədə spekulyasiya etmə.
3. Günah, məsuliyyət, "kim səhv etdi" kimi HƏR HANSI hüquqi qiymətləndirmə İFADƏ ETMƏ. Yalnız hadisəni neytral və obyektiv təsvir et.
4. Mətni Azərbaycan dilində, 2-3 cümlə olmaqla, qısa və dəqiq yaz."""

        # Naməlum/None olan faktları kənarlaşdırırıq ki, model onları nəzərə almasın
        filtered_facts = {k: v for k, v in facts.items() if v is not None}
        prompt = f"Hadisə faktları:\n{json.dumps(filtered_facts, ensure_ascii=False, indent=2)}\n\nBu faktlara əsasən yuxarıdakı qaydalara uyğun obyektiv hesabat mətni yarat."
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        # Request-ə timeout əlavə edirik (10 saniyə)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 10.0}
        )
        
        narrative_text = response.text.strip()
        
        result = {
            "text": narrative_text,
            "generatedBy": model_name,
            "disclaimer": "Bu təsvir avtomatik yaradılıb, yalnız mövcud sensor/vizual məlumatlara əsaslanır, hüquqi məsuliyyət və ya günah təyinatı deyil.",
            "success": True
        }
        
        _set_to_cache(cache_key, result)
        return result
        
    except Exception as e:
        print(f"[Narrative] xəta — səbəb: {str(e)}")
        return default_response
