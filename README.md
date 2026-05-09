# Baby Cry Classifier

Webová aplikace pro automatickou klasifikaci dětského pláče pomocí metod 
strojového učení. Aplikace vznikla jako implementační výstup bakalářské práce 
na Vysoké škole ekonomické v Praze.

## Popis

Aplikace analyzuje nahrávky dětského pláče a klasifikuje je do pěti kategorií:
hlad, nepohodlí, únava, bolest břicha a potřeba odříhnutí. Klasifikace je 
založena na extrakci MFCC příznaků a modelu Random Forest trénovaném na 
datasetu Donate-a-Cry.

Aplikace má experimentální povahu a není určena pro lékařskou diagnostiku. 
Výsledky klasifikace je nutné interpretovat s ohledem na limity použitého 
datasetu. Experimentální část práce je dostupná v repozitáři 
[baby-cry-classification-research](https://github.com/vojtechpardubsky/baby-cry-classification-research).

## Online demo

Aplikace je veřejně dostupná na adrese:  
https://baby-cry-app.onrender.com/

Při první návštěvě po delší době nečinnosti může dojít k prodlevě 30–60 sekund 
způsobené studeným startem serverové instance.

## Použité technologie

- Python, FastAPI — serverová část a inference modelu
- librosa — extrakce akustických příznaků
- scikit-learn — model Random Forest
- FFmpeg — konverze audio formátů
- HTML, CSS, JavaScript — uživatelské rozhraní
- Docker — kontejnerizace aplikace

## Spuštění lokálně

### S Dockerem (doporučeno)

```bash
docker build -t baby-cry-app .
docker run -p 10000:10000 baby-cry-app
```

Aplikace bude dostupná na `http://localhost:10000`.

### Bez Dockeru

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 10000
```

Aplikace bude dostupná na `http://localhost:10000`.

## Použití

1. Otevřete aplikaci v prohlížeči.
2. Nahrajte audio soubor (WAV, MP3) nebo pořiďte záznam z mikrofonu (7 sekund).
3. Spusťte klasifikaci a zobrazte výsledek.

## Licence

MIT License — viz soubor LICENSE.
