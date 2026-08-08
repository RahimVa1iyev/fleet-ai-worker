# AI-Worker: Full Pipeline Production Audit

---

## HİSSƏ A — Pipeline-ın tam axını

### A1. `process_job` funksiyasının tam kodu (`main.py`, sətir 153-278)

```python
async def process_job(job, job_token):
    """BullMQ-dan gələn hər işi emal et"""
    data = job.data
    event_id   = data.get("eventId")
    driver_id  = data.get("driverId", "unknown")
    frame_key  = data.get("frameR2Key") or data.get("frameUrl", "").replace("r2://fleet-events/", "")
    accel_data = data.get("accelData", {})
    event_type = data.get("eventType", "HARSH_BRAKING")
    gps        = data.get("gps") or {}
    lat        = gps.get("lat")
    lng        = gps.get("lng")

    print(f"[Worker] İş başladı — eventId: {event_id}, type: {event_type}")

    try:
        # 1. Status PROCESSING
        update_event_processing(event_id)

        # 2. R2-dən frame endir
        image_bytes = download_frame(frame_key)

        # 3. YOLO inference
        detections = run_inference(image_bytes)

        # 4. Enrichment (Weather & Road)
        weather_info = None
        road_info = None
        if lat is not None and lng is not None:
            try:
                weather_info = get_weather(float(lat), float(lng))
                road_info = get_road_info(float(lat), float(lng))
            except Exception as e:
                print(f"[Worker] Enrichment xətası (uduldu): {e}")
        else:
            print(f"[Worker] GPS tapılmadı — eventId: {event_id}, enrichment atlanır")

        # 5. Risk hesabla
        score, severity = calculate_risk(
            detections, accel_data, event_type,
            current_speed_kmh=gps.get("speed"),
            speed_limit_kmh=road_info.get("speedLimitKmh") if road_info else None,
        )

        # 6. Xülasə yarat
        summary = build_summary(detections, event_type, severity, score)

        # 7. Narrative Generation
        narrative_result = None
        try:
            facts = {
                "eventType": event_type,
                "speedKmhBefore": data.get("speedKmhBefore"),
                "speedKmhAfter": data.get("speedKmhAfter"),
                "gForce": accel_data.get("gForce") if accel_data else None,
                "detections": detections,
                "weather": weather_info,
                "roadType": road_info.get("roadType") if road_info else None,
                "speedLimitKmh": road_info.get("speedLimitKmh") if road_info else None,
                "timeOfDay": data.get("timeOfDay")
            }
            narrative_result = generate_narrative(facts)
        except Exception as e:
            print(f"[Worker] Narrative xətası (uduldu): {e}")

        ai_result = {
            "detections": detections,
            "riskScore":  score,
            "summary":    summary,
            "weather":    weather_info,
            "roadInfo":   road_info,
            "narrative":  narrative_result
        }

        # 8. DB-yə yaz
        update_event_completed(event_id, ai_result, severity, score, weather_info, road_info, narrative_result)

        # 9. FCM push
        fcm_token = get_driver_fcm_token(event_id)
        if fcm_token:
            send_fcm_notification(fcm_token, event_id, driver_id, severity, event_type)

    except Exception as e:
        error_msg = str(e)
        print(f"[Worker] XƏTA: {error_msg}")
        update_event_failed(event_id, error_msg)
        raise  # BullMQ retry üçün xətanı yenidən at
```

**Qeydlər:**
- Kod sətirləmə: Download Frame → YOLO → Enrichment → Risk → Summary → Narrative → DB write → FCM
- Frame endirimi (download_frame) xarici try/except-siz işləyir — hər hansı istisna birbaşa yuxarı `except Exception` bloğuna düşür.
- YOLO inference (run_inference) xarici try/except-siz işləyir — eyni.
- Enrichment try/except ilə qorunub, xəta halında `None` olaraq qalır.
- Narrative try/except ilə qorunub — qalan axın davam edir.
- Son `except Exception` bloku xətanı DB-yə yazır (`update_event_failed`) və BullMQ retry üçün yenidən atır (`raise`).

**RİSK SƏVİYYƏSİ: orta** — əsas axın mühafizəlidir, amma aşağıdakı bölmələrdə daha xüsusi problemlər var.

---

### A2. BullMQ consumer loop

```python
worker = Worker(
    QUEUE_NAME,      # = "ai-analysis"
    process_job,
    {
        "connection": {
            "host": os.environ.get("REDIS_HOST", "redis"),
            "port": int(os.environ.get("REDIS_PORT", 6379)),
        },
        "concurrency": int(os.environ.get("WORKER_CONCURRENCY", 2)),
        "stalledInterval": 30000,   # 30 saniyədə bir stalled job yoxlaması
        "maxStalledCount": 2,       # 2 dəfə stalled sayıldıqdan sonra FAILED
    }
)
```

- Queue adı: `"ai-analysis"` (sabitlənmiş, env-dən deyil)
- Default concurrency: 2 (paralel job sayı; `WORKER_CONCURRENCY` env ilə konfiqurasiya olunur)
- Stalled job mexanizmi mövcuddur: 30 saniyəlik interval, maksimum 2 dəfə stalled → FAILED

**RİSK SƏVİYYƏSİ: aşağı** — consumer konfiqurasiyası adekvat görünür.

---

## HİSSƏ B — YOLO nəticələrinin düzgün istifadəsi

### B3. `services/yolo.py`-nin tam analizi

**Confidence threshold:** 0.25 — sətir 61-də `if confidence < 0.25: continue` filtri var.

**NMS (Non-Max Suppression):** Klassik NMS tətbiq edilmir. Bunun əvəzinə sadələşdirilmiş "unikal saxla" məntiqi var (sətir 71-77):
```python
seen = {}
for d in detections:
    key = d["object"]
    if key not in seen or d["confidence"] > seen[key]["confidence"]:
        seen[key] = d
return list(seen.values())
```
Bu o deməkdir ki, hər class-dan **yalnız ən yüksək confidence-lı bir nümunə** saxlanılır. Real NMS-dən fərqli olaraq, eyni class-dan çox obyekt (məs. kadrda 3 nəfər) **sayılmır**. Bu, score hesablamasını həqiqi vəziyyətdən fərqli edə bilər.

**⚠️ TAPILDI: DEAD CODE — yolo.py-nin ikinci yarısı**
`services/yolo.py` faylında `run_inference` funksiyasının 77-ci sətirindən sonra — `return list(seen.values())` sonrasında — heç bir şərtlə əhatələnməmiş, buna görə də **heç vaxt icra olunmayan** ikinci bir implementasiya mövcuddur (sətir 78-115):
```python
    return list(seen.values())
    """
    Frame-i analiz et, aşkarlamaları qaytar.
    ...
    """
    session = get_session()
    ...
    return list(seen.values())
```
Bu, funksiyadan `return` etdikdən sonra gələn kod bloğudur. Bir docstring kimi görünsə də, `"""` arasındakı tam Python kodu mövqeyindədir. Syntax xətası doğurmur (Python-da `return`-dan sonrakı sətirləri sadəcə icra etmir), lakin ciddi konfuziya yaradır.

**Class label uyğunluğu:** YOLO `COCO_CLASSES` siyahısındakı adlar `calculate_risk`-dəki siyahılarla tam uyğundur:
- `high_risk_objects`: `"person"` ✅, `"bicycle"` ✅, `"motorcycle"` ✅, `"child"` ❌ (COCO-da `"child"` class-ı yoxdur, yalnız `"person"` var — `child` heç vaxt match olmayacaq)
- `medium_risk_objects`: `"car"` ✅, `"truck"` ✅, `"bus"` ✅, `"traffic light"` ✅, `"stop sign"` ✅

**⚠️ TAPILDI: `"child"` class-ı heç vaxt match olmur** — COCO-da belə bir class yoxdur. Bu sözün `high_risk_objects`-ə əlavə olunması risk hesablamasında heç bir əlavə xal qazandırmır.

**RİSK SƏVİYYƏSİ: orta** — dead code çaşqınlıq yarada bilər, `child` class-ı isə işləmir amma ciddi zərər vermir.

---

### B4. Detections siyahısı boş olduqda

`calculate_risk`-in ikinci bloku (objekt xalı):
```python
object_score = 0
for d in detections:   # boş siyahı — döngü işə düşmür
    ...
score += min(object_score, 40)  # += min(0, 40) = 0
```
Boş `detections` siyahısı halında `object_score=0` qalır, heç bir xəta atmır. Pipeline normal davam edir.

**RİSK SƏVİYYƏSİ: aşağı** — bu halda düzgün davranış var.

---

## HİSSƏ C — Risk hesablamasının edge-case-ləri

### C5. `accel_data` None və ya boş `{}` olduqda

```python
g_force = accel_data.get("gForce", 0) if accel_data else 0
```
- `accel_data = None` → `if accel_data` False olur → `g_force = 0` ✅
- `accel_data = {}` → `accel_data` truthy-dir → `.get("gForce", 0)` → `g_force = 0` ✅
- `accel_data = {"gForce": None}` → `.get("gForce", 0)` → `g_force = None` → **sonrakı müqayisədə** `None >= 1.0` TypeError! 

**⚠️ TAPILDI: `gForce: null` halı xəta verir** — əgər `accelData.gForce` əlan `null` olaraq göndərilibsə (məs. sensor məlumatı ötürülübsə amma dəyər yoxdursa), Python `TypeError: '>=' not supported between instances of 'NoneType' and 'float'` atır. Bu, `update_event_failed` çağırılması ilə nəticələnir.

**RİSK SƏVİYYƏSİ: orta** — real hardware-dən `gForce: null` gəlmə ehtimalı mövcuddur.

---

### C6. `gps.get("speed")` None olduqda (SPEEDING üçün)

```python
if current_speed_kmh and speed_limit_kmh and speed_limit_kmh > 0:
    ...
else:
    score += 20  # fallback
```
`current_speed_kmh = None` → `if None and ...` → False → fallback `score += 20`. Xəta atmır ✅.

**RİSK SƏVİYYƏSİ: aşağı** — düzgün davranış var.

---

### C7. `road_info` tamamilə None olduqda

Enrichment xəta verdikdə `road_info = None` qalır. Sonra:
```python
speed_limit_kmh=road_info.get("speedLimitKmh") if road_info else None,  # ✅
...
road_info.get("roadType") if road_info else None,                         # ✅ DB yazımında
```
`None` guard-lar mövcuddur. Pipeline xətasız davam edir, `roadType` və `speedLimitKmh` DB-yə `NULL` yazılır.

**RİSK SƏVİYYƏSİ: aşağı** — düzgün idarə olunur.

---

### C8. Naməlum `event_type` olduqda

```python
score += base_scores.get(event_type, 20)  # default=20
```
Heç bir exception atmır, sadəcə 20 baza xalı ilə davam edir ✅. `build_summary` funksiyasında:
```python
event_name = event_names.get(event_type, event_type)  # tanınmayan növü olduğu kimi göstərir
```
Bu da xəta atmır ✅.

**RİSK SƏVİYYƏSİ: aşağı** — naməlum event tipləri sakit şəkildə idarə olunur.

---

## HİSSƏ D — Xəta idarəetməsi, retry, timeout

### D9. Frame R2-dən endirilə bilmədikdə

`services/storage.py` faylında `download_frame`:
```python
def download_frame(r2_key: str) -> bytes:
    client = get_s3_client()
    bucket = os.environ["R2_BUCKET"]
    response = client.get_object(Bucket=bucket, Key=r2_key)
    return response["Body"].read()
```
Heç bir try/except yoxdur, timeout ayarı da yoxdur. R2 xəta versə (fayl yoxdursa `NoSuchKey`, şəbəkə xətası), exception `process_job`-un böyük `except Exception` blokuna düşür → `update_event_failed` çağırılır → `raise` → BullMQ retry.

**⚠️ TAPILDI: boto3 timeout ayarı yoxdur** — `Config()` obyektinə `connect_timeout`, `read_timeout` parametrləri ötürülməyib. Şəbəkə problem yaşadıqda `download_frame` sonsuz müddət asılı qala bilər, worker thread-i həmin job üçün bloklanır.

**⚠️ TAPILDI: `frame_key` None ola bilər** — əgər job data-sında nə `frameR2Key`, nə də `frameUrl` yoxdursa, `frame_key = None or "".replace(...) = ""`. Boş sətir ilə S3 sorğusu ya xəta verir, ya da səhv fayl endirər.

**RİSK SƏVİYYƏSİ: yüksək** — timeout yoxluğu production-da worker-in asılı qalmasına səbəb ola bilər.

---

### D10. YOLO inference xəta versə

`run_inference` heç bir try/except olmadan `process_job`-dan çağırılır. Korrupt şəkil, model yüklənmə xətası, yaddaş xətası — hamısı yuxarıdakı `except Exception` blokuna düşür → DB-yə FAILED yazılır → retry. Bu davranış məntiqlidir, lakin YOLO model faylı yoxdursa (`session = None`) bütün işlər ardıcıl FAILED olacaq.

**RİSK SƏVİYYƏSİ: orta** — worker səviyyəsində xəta idarəsi var, model faylı olmazsa tam sistem dayanır.

---

### D11. Gemini narrative API xəta versə

```python
narrative_result = None
try:
    narrative_result = generate_narrative(facts)
except Exception as e:
    print(f"[Worker] Narrative xətası (uduldu): {e}")
```
Narrative bloku try/except ilə qorunub. Xəta halında `narrative_result = None` qalır, amma DB yazımı (`update_event_completed`) **yenə də çağırılır** — score/severity DB-yə yazılır. ✅

Əlavə olaraq, `narrative.py` özündə də daxili try/except var və xəta halında `default_response = {"text": None, ..., "success": False}` qaytarır. Yəni iki qat qoruma mövcuddur.

Timeout: `genai.Client(api_key=..., http_options=types.HttpOptions(timeout=10000))` — 10 saniyəlik timeout konfiqurasiya edilib ✅.

**RİSK SƏVİYYƏSİ: aşağı** — narrative xətaları düzgün idarə olunur.

---

### D12. Retry məntiqi

`bullmq.service.ts`-dəki `aiQueue` konfiqurasiyası:
```typescript
this.aiQueue = new Queue(QUEUE_NAMES.AI_ANALYSIS, {
    connection: redisConnection,
    defaultJobOptions: {
        attempts: 3,                                    // Cəmi 3 cəhd
        backoff: { type: 'exponential', delay: 5000 }, // İlk retry 5s, sonra 10s, 20s
        removeOnComplete: 100,
        removeOnFail: 50,
    },
});
```
- 3 cəhd, eksponensial backoff (5s, 10s, 20s).
- `removeOnFail: 50` — yalnız son 50 FAILED job saxlanılır, qalanları Redis-dən silinir.

**⚠️ TAPILDI: `removeOnFail: 50` monitoring riskini artırır** — köhnə FAILED job-lar avtomatik silinir. Əgər monitoring sistemi yoxdursa, geçmiş xətaları retrospektiv araşdırmaq mümkün olmaya bilər (aşağıdakı D13-ə bax).

**RİSK SƏVİYYƏSİ: aşağı** — retry məntiqi adekvat konfiqurasiya olunub.

---

### D13. Dead Letter / FAILED monitorinqi

FAILED job-lar üçün `main.py`-da `on_failed` handler var:
```python
async def on_failed(job, err, prev_state):
    attempts = job.attempts_made if hasattr(job, 'attempts_made') else 0
    print(f"[Worker] İş uğursuz — eventId: {job.data.get('eventId')}, cəhd: {attempts}, xəta: {err}")
    if attempts >= 3:
        event_id = job.data.get("eventId")
        if event_id:
            update_event_failed(event_id, str(err))
            print(f"[Worker] FAILED yazildi — eventId: {event_id}")
```

**⚠️ TAPILDI: Monitoring yalnız `print` ilə məhdudlaşır** — Heç bir xarici alert sistemi (Sentry, PagerDuty, webhook) bağlı deyil. FAILED job-lar DB-yə `errorReason` ilə yazılır, lakin passiv izləmə (aktiv kömandanın xəbərdarlıq alması) mövcud deyil. Ayrıca, `removeOnFail: 50` Redis-dəki FAILED job-ları siləcəyi üçün BullMQ dashboard-dan (məs. Bull Board) geçmiş xətalara baxmaq mümkünsüz olacaq.

**RİSK SƏVİYYƏSİ: orta** — xətalar DB-yə yazılır amma aktiv monitorinq yoxdur.

---

## HİSSƏ E — Race condition / idempotentlik

### E14. Eyni eventId üçün ikinci dəfə job gəlsə

`update_event_processing` funksiyası sadəcə `UPDATE ... WHERE id = %s` edir — idempotent deyil, lakin zərərsizdir. İkinci job ilk job-u "izləyəcək" — bu halda iki paralel YOLO analizi icra olacaq, ikincisi DB-ni öz nəticəsi ilə yazacaq.

**⚠️ TAPILDI: İdempotentlik mexanizmi yoxdur** — Worker eyni event-i iki dəfə emal etməyə çalışmaq üçün heç bir "əgər bu event artıq COMPLETED/PROCESSING-dirsə, atla" yoxlaması yoxdur. İki eyni job queue-ya düşərsə, ikisi də tam icra olunacaq. Son yazan kazanar, amma bu əgər YOLO nəticələri fərqlidirsə (iki paralel inference zamanı race condition) qeyri-deterministik olacaq.

**RİSK SƏVİYYƏSİ: orta** — eyni event-in iki dəfə çağırılma ehtimalı (şəbəkə retry) mövcuddur.

---

### E15. Worker prosesi ortada çöksə

`stalledInterval: 30000` və `maxStalledCount: 2` konfiqurasiya olunub. BullMQ, aktiv icra olunan job-ları `lock` mexanizmi ilə izləyir. Worker həmin job üçün lock-ı yeniləməsə (process çökdüksə), BullMQ 30 saniyəlik intervalda lock-ın bitdiyini müəyyən edir, job-u "stalled" sayır. 2 dəfə stalled olandan sonra job FAILED kimi qeyd edilir — yeni worker tərəfindən **yenidən götürülmür**, birbaşa FAILED-ə keçir.

**⚠️ TAPILDI: Stalled job RETRY edilmir, birbaşa FAILED olur** — `maxStalledCount: 2` konfiqurasiyasına görə, 2 dəfə stalled sayılan job FAILED-ə keçir. Bu, əvvəlki sessiyalarda müşahidə olunan "stuck" davranışının səbəbidir: Ctrl+C ilə öldürülmüş worker-in icra etdiyi job lock-ı itirəcək, stalled sayılacaq, nəhayət FAILED olacaq. `removeOnFail: 50` isə həmin job-u Redis-dən siləcəyi üçün vizual olaraq "itmiş" görünəcək.

**RİSK SƏVİYYƏSİ: orta** — stalled→FAILED davranışı anlaşılan, lakin bunu izləmək üçün monitorinq lazımdır.

---

## ÜMUMİ NƏTİCƏ — Production-a çıxmazdan əvvəl prioritetləşdirilmiş siyahı

### 🔴 YÜKSƏK RİSK — Həll edilməlidir

| # | Problem | Fayl | Sətir |
|---|---------|------|-------|
| 1 | **boto3 R2 download-da timeout yoxdur** — şəbəkə problemi zamanı worker thread sonsuz asılı qalır | `services/storage.py` | 6-13 |
| 2 | **`gForce: null` TypeError xətası** — `accelData.gForce` null gəldikdə `NoneType >= float` xətası baş verir, job FAILED olur | `main.py` | 116 |
| 3 | **`frame_key` boş string ola bilər** — `frameR2Key` yoxdursa, boş sətir ilə S3 sorğusu göndərilir | `main.py` | 158 |

### 🟡 ORTA RİSK — Baxılmalıdır

| # | Problem | Fayl | Sətir |
|---|---------|------|-------|
| 4 | **Yolo.py-də dead code** — `return`-dan sonra 37 sətir icra olunmayan kod var; gələcəkdə çaşqınlıq yaradacaq | `services/yolo.py` | 78-115 |
| 5 | **İdempotentlik yoxdur** — eyni eventId-li ikinci job gəlsə, iki tam emal icra olunur | `main.py` | 167-169 |
| 6 | **Aktiv monitorinq/alert yoxdur** — FAILED job-lar yalnız DB-yə yazılır, komanда real vaxtda xəbərdar olunmur | `main.py` | 264-271 |
| 7 | **Stalled job FAILED olur, retry edilmir** — Ctrl+C zamanı işi "itmiş" kimi görünür | `main.py` | 258-261 |

### 🟢 AŞAĞI RİSK — Uyğun vaxtda baxılmalıdır

| # | Problem | Fayl | Sətir |
|---|---------|------|-------|
| 8 | **`"child"` class-ı heç vaxt match olmur** — COCO-da bu class yoxdur, `high_risk_objects`-dən çıxarıla bilər | `main.py` | 102 |
| 9 | **`removeOnFail: 50`** — köhnə FAILED job-lar Redis-dən silinir, retrospektiv analiz çətinləşir | `bullmq.service.ts` | 42 |
| 10 | **NMS tam deyil** — kadrda 3 nəfər varsa, yalnız ən yüksək confidence-lı 1 nəfər sayılır | `services/yolo.py` | 71-77 |
