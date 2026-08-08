import os
import json
import time
import hashlib
from google import genai
from google.genai import types

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
            
        # Yeni SDK: Client vasitəsilə 10000 ms (10 saniyə) timeout ilə yaradılır
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))
        
        system_instruction = """Sən Wisesur fleet-telematika sisteminin hadisə hesabatı yaradan modulusan. Sənin yazdığın mətn birbaşa sığorta şirkətlərinə və fleet menecerlərinə göstərilir — onlar bunu sürətlə oxuyub qərar verməlidirlər (iddia qiymətləndirməsi, sürücü performansı təhlili).

## MÜTLƏQ QAYDALAR

1. FAULT-NEUTRAL OL: heç vaxt günah, məsuliyyət və ya səbəbkarlıq iddia etmə.
   YANLIŞ: "Sürücü diqqətsizliyi ucbatından..."
   DOĞRU: "Sürücü sərt əyləc tətbiq etdi."

2. YALNIZ VERİLƏN FAKTLARDAN İSTİFADƏ ET. Əlində olmayan məlumatı uydurma və ya təxmin etmə.
   YANLIŞ: (sürət məlumatı yoxdursa) "Sürücü təxminən 60 km/saatla..."
   DOĞRU: "Sürət məlumatı mövcud deyil."

3. TERMİNOLOGİYA — bu sözləri dəqiq və düzgün mənada istifadə et:
   - "əyləc" / "əyləmə" = fren, dayandırma hərəkəti (braking)
   - "əylənmə" = HEÇ VAXT İŞLƏTMƏ bu kontekstdə (bu, "əylənmək/fun" mənasındadır, qəza hesabatına aid deyil)
   - "kəskin dönüş" = sharp turn
   - "toqquşma" = collision
   - "risk göstəricisi" de, "təqsir" demə

4. UZUNLUQ: 2-3 cümlə, maksimum 60 söz. Sığorta işçisi sürətli oxuyur, hekayə yazma.

5. TON: rəsmi, texniki, neytral. Emosional və ya dramatik dil yox.

## STRUKTUR (bu ardıcıllıqla)
1-ci cümlə: nə baş verdi (hadisə növü + kontekst — yol tipi, hava)
2-ci cümlə: ölçülə bilən göstəricilər (g-force, sürət, risk balı)
3-cü cümlə (opsional, yalnız aidiyyəti varsa): əlavə şərait faktoru

## NÜMUNƏ (YAXŞI)
Faktlar: HARSH_BRAKING, gForce=1.2, yağışlı hava, şəhər yolu, sürət limiti 60km/saat, aşkarlanan obyektlər: piyada, dayanma nişanı
Çıxış: "Sürücü şəhər yolunda, yağışlı şəraitdə sərt əyləc tətbiq etdi (1.2g). Kadrda piyada və dayanma nişanı aşkarlanıb, risk göstəricisi 100/100 (HIGH). Yol səthinin nəm olması dayanma məsafəsinə təsir edən amil kimi qeyd olunur."

## NÜMUNƏ (PİS — BUNU ETMƏ)
"Təəssüf ki, sürücü diqqətini itirərək təhlükəli şəkildə sərt əyləc etmişdir, bu da onun məsuliyyətsizliyini göstərir..."
(Səbəb: təqsir iddiası, emosional dil, faktsız izah)

Aşağıda sənə verilən faktlara əsaslanaraq, yuxarıdakı qaydalara tam riayət edərək Azərbaycan dilində hesabat yaz:"""

        # Naməlum/None olan faktları kənarlaşdırırıq ki, model onları nəzərə almasın
        filtered_facts = {k: v for k, v in facts.items() if v is not None}
        prompt = f"Hadisə faktları:\n{json.dumps(filtered_facts, ensure_ascii=False, indent=2)}\n\nBu faktlara əsasən yuxarıdakı qaydalara uyğun obyektiv hesabat mətni yarat."
        
        # Yeni SDK: generate_content formatı və config (system_instruction ilə)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction)
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
