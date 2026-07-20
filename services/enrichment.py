import os
import time
import requests

# Sadə in-memory cache
# Format: {"prefix_lat_lng": (data_dict, timestamp)}
_cache = {}
CACHE_TTL = 900  # 15 dəqiqə (saniyə ilə)
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "info@wisesur.com")
USER_AGENT = f"Wisesur-FleetTelematics/1.0 (contact: {CONTACT_EMAIL})"

def _get_cache_key(prefix: str, lat: float, lng: float) -> str:
    # 1km dəqiqlik üçün koordinatları 2 onluq rəqəmə yuvarlaqlaşdırırıq
    return f"{prefix}_{round(lat, 2)}_{round(lng, 2)}"

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

def get_weather(lat: float, lng: float) -> dict:
    default_response = {
        "condition": None,
        "temperatureC": None,
        "precipitation": None,
        "source": "unavailable"
    }

    try:
        cache_key = _get_cache_key("weather", lat, lng)
        cached_data = _get_from_cache(cache_key)
        if cached_data:
            return cached_data

        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key:
            print("[Enrichment] xəta — funksiya: get_weather, səbəb: OPENWEATHER_API_KEY tapılmadı")
            return default_response

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={api_key}&units=metric"
        headers = {"User-Agent": USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        
        condition = data["weather"][0]["main"] if data.get("weather") else "Unknown"
        temperature_c = float(data["main"]["temp"])
        
        # Yağıntı olub-olmamasını həm "rain/snow" keylərindən, həm də hava vəziyyətinin adından yoxlayırıq
        condition_lower = condition.lower()
        precipitation = ("rain" in data or "snow" in data or 
                         condition_lower in ["rain", "snow", "drizzle", "thunderstorm"])

        result = {
            "condition": condition,
            "temperatureC": temperature_c,
            "precipitation": precipitation,
            "source": "openweathermap"
        }

        _set_to_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"[Enrichment] xəta — funksiya: get_weather, səbəb: {str(e)}")
        return default_response

def get_road_info(lat: float, lng: float) -> dict:
    default_response = {
        "roadType": None,
        "speedLimitKmh": None,
        "source": "unavailable"
    }

    try:
        cache_key = _get_cache_key("road", lat, lng)
        cached_data = _get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Ən yaxın 50 metr radiusda olan yolu (highway) tapmaq üçün Overpass API sorğusu
        query = f"""
        [out:json];
        way(around:50,{lat},{lng})[highway];
        out tags;
        """
        
        url = "https://overpass-api.de/api/interpreter"
        headers = {"User-Agent": USER_AGENT}
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, data={"data": query}, timeout=10)
                response.raise_for_status()
                data = response.json()
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise e
        elements = data.get("elements", [])

        if not elements:
            return default_response

        # Yollar üçün prioritetlərin müəyyən edilməsi
        high_priority_highways = {"primary", "secondary", "tertiary", "residential", "trunk", "motorway"}
        low_priority_highways = {"footway", "path", "service", "pedestrian"}
        
        best_road = None
        best_priority = -1 # -1: tapılmayıb, 0: aşağı prioritet (piyada), 1: avtomobil yolu
        
        for el in elements:
            tags = el.get("tags", {})
            highway_type = tags.get("highway")
            
            if not highway_type:
                continue
                
            if highway_type in high_priority_highways:
                # Avtomobil yolu tapıldı, prioritet ən yüksəkdir, dərhal seçirik
                best_road = el
                best_priority = 1
                break
            elif highway_type not in low_priority_highways and best_priority < 1:
                # Nə avtomobil yoludur, nə də piyada (məsələn, 'unclassified'). Yenə də piyadadan yaxşıdır.
                best_road = el
                best_priority = 0.5
            elif best_priority < 0:
                # Yalnız piyada/xidmət yolu tapmışıqsa, hələlik bunu qəbul edirik
                best_road = el
                best_priority = 0

        if not best_road:
            return default_response

        tags = best_road.get("tags", {})
        road_type = tags.get("highway", "unknown")
        maxspeed_str = tags.get("maxspeed")
        
        speed_limit_kmh = None
        if maxspeed_str and maxspeed_str.isdigit():
            speed_limit_kmh = int(maxspeed_str)

        result = {
            "roadType": road_type,
            "speedLimitKmh": speed_limit_kmh,
            "source": "osm"
        }

        _set_to_cache(cache_key, result)
        return result

    except Exception as e:
        print(f"[Enrichment] xəta — funksiya: get_road_info, səbəb: {str(e)}")
        return default_response
