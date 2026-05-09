# Baby Cry Classifier

Webová aplikace pro automatickou klasifikaci dětského pláče pomocí metod 
strojového učení. Aplikace vznikla jako součást bakalářské práce na téma 
automatické klasifikace dětského pláče.

## Popis

Aplikace analyzuje nahrávky dětského pláče a klasifikuje je do pěti kategorií:
hlad, nepohodlí, únava, bolest břicha a potřeba odříhnutí. Klasifikace je 
založena na extrakci MFCC příznaků a modelu Random Forest trénovaném na 
datasetu Donate-a-Cry.

Aplikace má experimentální povahu a slouží primárně k ověření funkčnosti 
navrženého klasifikačního systému. Výsledky klasifikace je nutné interpretovat 
s ohledem na limity použitého datasetu popsané v přiložené práci.

## Použité technologie

- Python, FastAPI — serverová část a inference modelu
- librosa — extrakce akustických příznaků
- scikit-learn — model Random Forest
- FFmpeg — konverze audio formátů
- HTML, CSS, JavaScript — uživatelské rozhraní
- Docker — kontejnerizace aplikace

## Spuštění lokálně

### Bez Dockeru

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Aplikace bude dostupná na `http://localhost:8000`.

### S Dockerem

```bash
docker build -t baby-cry-classifier .
docker run -p 10000:10000 baby-cry-classifier
```

Aplikace bude dostupná na `http://localhost:10000`.

## Použití

1. Otevřete aplikaci v prohlížeči.
2. Nahrajte audio soubor ve formátu WAV nebo MP3, nebo pořiďte záznam 
   přímo z mikrofonu (maximálně 7 sekund).
3. Spusťte klasifikaci a zobrazte výsledek.

## Licence

MIT License — viz soubor LICENSE.
