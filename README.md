# Odak Takibi Streamlit MVP

Bu klasor, DAiSEE tabanli odak tahmin modelini kullanan bagimsiz Streamlit MVP uygulamasidir. Egitim kodlari, notebooklar ve dataset dosyalari bu uygulamayi calistirmak icin gerekli degildir.

## Uygulamanin Amaci

Uygulama, kullanicidan alinan yuz goruntulerinden odak seviyesini tahmin etmeyi hedefler.

Temel akis:

1. Kullanici bir goruntu yukler veya kameradan snapshot alir.
2. Model goruntuyu `low_focus` veya `focused` siniflarina gore yorumlar.
3. `focused` sinifi olasiligi 0-100 arasi odak skoruna donusturulur.
4. Son snapshot skorlarinin ortalamasi alinarak daha stabil bir ortalama odak skoru uretilir.
5. Ortalama skora gore `Dusuk Odak`, `Ortalama Odak` veya `Yuksek Odak` durumu gosterilir.

## Klasor Yapisi

```text
streamlit_mvp/
├── app2.py
├── mvp_focus.py
├── models/
│   └── best_model.pt
├── reports/
└── requirements.txt
```

Dosyalar:

- `app2.py`: Streamlit arayuzu.
- `mvp_focus.py`: Model yukleme, tahmin, yuz algilama ve rolling window kodlari.
- `models/best_model.pt`: Egitilmis PyTorch model checkpoint dosyasi.
- `reports/`: Canli kamera sonuclarinin kaydedildigi klasor.
- `requirements.txt`: Uygulama icin gerekli Python paketleri.

## Model Mantigi

MVP'de EfficientNet-B0 tabanli binary classification modeli kullanilir.

Egitim etiketi:

```text
Engagement 0/1 -> low_focus
Engagement 2/3 -> focused
```

Model softmax ile iki olasilik uretir:

```text
P(low_focus), P(focused)
```

Anlik odak skoru:

```text
Anlik odak skoru = P(focused) * 100
```

Ornek:

```text
P(focused) = 0.73
Anlik odak skoru = 73
```

## Ortalama Odak Skoru

Tek kare tahminleri gurultulu olabilir. Bu nedenle uygulama son N snapshot skorunu saklar ve ortalamasini alir.

```text
Ortalama odak skoru = son N anlik odak skorunun ortalamasi
```

Varsayilan pencere boyutu:

```text
N = 10
```

Bu, tek bir kotu kare yuzunden durumun aniden degismesini azaltir.

## Durum Seviyeleri

Ortalama odak skoruna gore uc seviye uretilir:

```text
0-49   -> Dusuk Odak
50-59  -> Ortalama Odak
60-100 -> Yuksek Odak
```

Durum gecmisinde saat bilgisiyle aciklama gosterilir:

```text
Saat 13:45:10 civarinda odak seviyeniz ortalama seviyedeydi.
```

Saat bilgisi Turkiye saatine gore uretilir.

## Upload ve Kamera Farki

Upload edilen gorseller:

- Dogrudan modele verilir.
- Face detection uygulanmaz.
- Hazir crop edilmis test gorselleri icin daha uygundur.

Kamera snapshot'lari:

- Once OpenCV ile yuz algilama yapilir.
- Algilanan en buyuk yuz bolgesi crop edilir.
- Modele sadece yuz crop'u verilir.

Bu ayrim bilincli olarak yapilmistir. Webcam goruntusu genelde yuz disinda arka plan, ekran, masa ve isik gibi ek bilgiler icerir. Modelin egitim verisi yuz frame'lerine daha yakin oldugu icin kamera tarafinda yuz crop kullanilir.

## Uygulama Sekmeleri

### Tek goruntu

Bu sekmede:

- Bir veya daha fazla goruntu yuklenebilir.
- Kameradan tek snapshot alinabilir.
- Kamera ve skorlar yan yana gosterilir.

### Canli kamera

Bu sekme her saniye bir snapshot alir.

Akis:

```text
kamera frame'i -> yuz algilama -> yuz crop -> model tahmini -> rolling window -> durum
```

Canli kamera sonuclari su dosyaya kaydedilir:

```text
reports/live_camera/app_live_predictions.csv
```

### Odak gecmisi

Bu sekmede:

- Anlik odak skoru grafigi
- Ortalama odak skoru grafigi
- Son tahmin tablosu
- Saatli durum gecmisi

gosterilir.

## Kurulum

Sanal ortami aktif et:

```powershell
cd focus-score-daisee-streamlit
.\.venv\Scripts\Activate.ps1
```

Gerekli paketleri kur:

```powershell
pip install -r focus-score-daisee-streamlit\requirements.txt
```

Model dosyasinin su konumda oldugundan emin ol:

```text
focus-score-daisee-streamlit/models/best_model.pt
```

## Calistirma

```powershell
cd focus-score-daisee-streamlit
streamlit run app2.py
```

## Kullanim

1. `Tek goruntu` sekmesinden gorsel yukleyerek veya kamera snapshot alarak test yap.
2. `Canli kamera` sekmesinden calisma suresini sec.
3. `Canli snapshot akisini baslat` butonuna bas.
4. Sonuclari `Odak gecmisi` sekmesinden incele.

## Sik Karsilasilan Durumlar

### Model dosyasi bulunamadi

`models/best_model.pt` dosyasinin gercekten var oldugunu kontrol et.

### Kamera acilamadi

Kameranin baska bir uygulama tarafindan kullanilmadigindan emin ol. Tarayici kamera izni ile OpenCV kamera erisimi farkli mekanizmalardir.

### Baslangicta skor yuksek olup sonra hizli dusuyor

Ilk karelerde rolling window henuz dolu olmadigi icin ortalama skor hizli degisebilir. Ayrica yuz crop'u her karede biraz farkli olursa model skoru oynayabilir. Daha stabil sonuc icin pencere boyutu artirilabilir.

## Sinirlar

Bu MVP klinik veya resmi dikkat olcum sistemi degildir. Model DAiSEE veri seti uzerinde egitildigi icin gercek webcam kosullarinda isik, kamera acisi, yuz crop kalitesi ve domain farki sonuclari etkileyebilir.
