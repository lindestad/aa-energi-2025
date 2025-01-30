# aa-energi-2025

**Project information in English:** [README-ENGLISH.md](README-ENGLISH.md)

----

# 1. MLP Sinus-Cosinus-regresjon

En rask demonstrasjon av bruk av **PyTorch** for å tilpasse en 2D sinus-cosinus-funksjon ved hjelp av to forskjellige Multi-Layer Perceptron (MLP)-arkitekturer:

- **Liten MLP** (moderat kapasitet)  
- **Stor MLP** (høy kapasitet, for å overtilpasse og fange opp fine detaljer)

![Sinus Cos Overfitting Demo](assets/img/1-sincos.png)

# **2. Tidsserieprognoser med små datasett – Krig mot autoregressoren**  

**_Vi forsøker å utfordre klassiske autoregressive modeller ved hjelp av dyp læring på et lite datasett—og det var en tung kamp._**  

![Prediksjonstidslinje](assets/img/2_predicted.png)  

----

# 1. MLP Sinus-Cosinus-regresjon

En rask demonstrasjon av bruk av **PyTorch** for å tilpasse en 2D sinus-cosinus-funksjon ved hjelp av to forskjellige Multi-Layer Perceptron (MLP)-arkitekturer:

- **Liten MLP** (moderat kapasitet)  
- **Stor MLP** (høy kapasitet, for å overtilpasse og fange opp fine detaljer)

![Sinus Cos Overfitting Demo](assets/img/1-sincos.png)

## Oversikt

1. **Data**  
   Vi har $(x, i)$-par og et mål $z = \sin(\cos(x)) + \sin(\cos(i))$.  
   - Inndata er lagret i `data/X_sincos.txt`
   - Målverdier er lagret i `data/y_sincos.txt`
   - Data levert av Å Energi.

2. **Modeller**  
   - **SmallMLP**: 2 → 100 → 100 → 1 (med ReLU-aktiveringer)  
   - **LargeMLP**: 2 → 100 → 500 → 500 → 100 → 1 (med ReLU-aktiveringer)

3. **Trening**  
   - Vi bruker **MSE Loss** og **Adam Optimizer** i PyTorch.  
   - **LargeMLP** er bevisst overparameterisert for å tilpasse (og til og med overtilpasse) dataene svært godt.

4. **Resultater**  
   - Vi sammenligner den sanne funksjonen med prediksjoner fra begge MLP-ene i en 3D-plot.

![Sinus Cos Overfitting Demo](assets/img/1-sincos.png)

> *Fra venstre til høyre*:  
> **(1)** Sann funksjon  
> **(2)** Prediksjoner fra liten MLP  
> **(3)** Prediksjoner fra stor MLP

## Kom i gang

1. **Installer avhengigheter**  
   ```bash
   pip install torch matplotlib numpy
   ```
2. **Kjør scriptet**  
   ```bash
   python 1-sincos.py
   ```
   - Juster hyperparametere (epoker, læringsrate) om ønskelig.

3. **Plotting**  
   Scriptet viser automatisk et 3D-overflateplot for å sammenligne prediksjonene.

## Lærdommer

- **Små vs. store modeller**: Et større nettverk kan tilnærme målfunksjonen svært nøyaktig, men kan overtilpasse dersom datagrunnlaget er begrenset.  
- **Visualisering**: 3D-overflateplott hjelper oss med å visuelt vurdere hvor godt modellen fanger opp den underliggende funksjonen.  
- **PyTorch**: Viser hvor enkelt det er å bygge og trene MLP-er på egendefinerte data med bare noen få linjer Python-kode.

----


# **2. Tidsserieprognoser med små datasett – Krig mot autoregressoren**  

**_Vi forsøker å utfordre klassiske autoregressive modeller ved hjelp av dyp læring på et lite datasett—og det var en tung kamp._**  

![Prediksjonstidslinje](assets/img/2_predicted.png)  

## **Oversikt**  

Dette prosjektet utforsker forskjellige tilnærminger til å forutsi en daglig tidsserie med kun **~4000 observasjoner** (etter å ha tatt hensyn til lag-funksjoner). Målet var å undersøke om moderne dyp læring—LSTMer og Transformere—kunne overgå klassiske statistiske metoder i en situasjon med lite data.  

Vi testet fire modeller:  

1. **Naiv AR**: Den enkleste baselinen—antar at dagens verdi vil være den samme som i går.  
2. **AR(5)**: En lineær autoregressiv modell som bruker de siste fem dagene for å forutsi neste dag.  
3. **LSTM**: Et rekurrent nevralt nettverk trent på sekvenser av 30 dager.  
4. **Transformer**: En selvoppmerksomhetsmodell som også bruker et 30-dagers vindu.  

### **Vant dyp læring?**  

Ikke denne gangen. Med bare noen få tusen datapunkter og kun tre eksogene variabler (`x1, x2, x3`), slet de nevrale nettverkene med å finne meningsfulle mønstre. De autoregressive modellene, spesielt AR(5), presterte betydelig bedre fordi:  

- Datasettet er **svært lite** (~4000 rader), noe som begrenser læringskapasiteten til dype modeller.  
- De eksogene variablene har **svak forklaringskraft**, noe som betyr at de ikke bidrar mye til prognosen.  
- Tidsserien i seg selv er **sterkt autoregressiv**, noe som betyr at tidligere verdier alene gir et sterkt prediktivt signal—noe de enklere AR-modellene håndterer godt.  

## **Endelige resultater**  

| Modell       | MAE  | MSE  |  
|-------------|------|------|  
| **Naiv AR**  | **2.626**  | **19.377**  |  
| **AR(5)**     | **2.466**  | **17.183**  |  
| LSTM         | 4.930  | 59.427  |  
| Transformer  | 5.853  | 70.126  |  

Både LSTM og Transformer ble klart slått av de naive og AR(5)-modellene. De dype læringsmodellene hadde nesten **dobbelt så høy MAE** og **tre til fire ganger høyere MSE**. En klar seier for den klassiske tilnærmingen i dette tilfellet.  

---

## **Viktige grafer**  

### **Prediksjonstidslinje**  

Denne grafen sammenligner faktiske og predikerte verdier over tid. Jo nærmere en modells prediksjoner følger de virkelige verdiene, desto bedre presterer den.  

📌 **Hva du bør se etter:**  
- Hvilke modeller ligger nærmest de faktiske verdiene? Her gjør de autoregressive modellene en langt bedre jobb.  
- Henger noen modeller konsekvent etter eller overpredikerer målet? Det er ingen systematisk forsinkelse, noe som indikerer at alt er satt opp riktig og at hyperparametrene er rimelige.  
- Hvor mye støy introduserer LSTM og Transformer sammenlignet med AR(5)? Svaret er betydelig støy og tilfeldige topper—datasettet er for lite til at nevrale nettverk kan skinne!  

![Prediksjonstidslinje](assets/img/2_predicted.png)  

### **Absolutt feil over tid**  

Denne grafen viser hvordan hver modells absolutte feil utvikler seg over tid. Den hjelper med å identifisere perioder hvor modellene sliter mest.  

📌 **Hva du bør se etter:**  
- Er det spesifikke tidsperioder hvor feilene øker kraftig? De største toppene sammenfaller med store bevegelser i $y$, og fordi datasettet ikke inneholder nok forklaringskraft, gjør LSTMer og Transformere store feil.  
- Gjør én modell konsekvent større feil enn de andre?  
- Viser dype læringsmodeller ustabil eller uforutsigbar atferd?  

![Feiltidslinje](assets/img/2_error_timeline.png)  

### **MAE- og MSE-sammenligning**  

Disse søylediagrammene gir en direkte numerisk sammenligning av hvor godt hver modell presterte.  

- **MAE (Mean Absolute Error)** viser gjennomsnittsstørrelsen på feilene på en intuitiv måte.  
- **MSE (Mean Squared Error)** gir større vekt til store feil, noe som gjør den mer følsom for ekstreme avvik.  

📌 **Hva du bør se etter:**  
- AR(5)-modellen oppnår lavest MAE og MSE—vinneren av denne utfordringen.  
- LSTM og Transformer har betydelig høyere feil, noe som viser at de sliter med det begrensede datasettet.  
- Den naive modellen presterer overraskende godt, noe som viser hvor sterkt autoregressiv tidsserien er.  

**MAE-sammenligning:**  
![MAE](assets/img/2-MAE.png)  

**MSE-sammenligning:**  
![MSE](assets/img/2-MSE.png)  

---

## **Hvordan kjøre koden**  

1. Installer avhengigheter:  
   ```bash
   pip install numpy pandas matplotlib torch scikit-learn
   ```
2. Kjør hovedskriptet:  
   ```bash
   python 2-tahps.py
   ```
3. Sjekk konsollutdata og genererte grafer.  

---

## **Konklusjon**  

Til tross for våre beste forsøk, **vant ikke dyp læring denne kampen**—men det er ikke overraskende. AR(5) og til og med den naive modellen presterte godt fordi tidligere verdier alene inneholdt nok prediktiv informasjon.  

Imidlertid, i et scenario med **mer data** og **sterkere eksogene variabler**, kunne LSTM og Transformer ha gjort det bedre. Foreløpig fremhever dette prosjektet en viktig lærdom innen tidsserieprognoser: **noen ganger er det enkleste også det beste.**  

Vil du eksperimentere? Prøv å legge til flere funksjoner, justere hyperparametere eller bruke forskjellige arkitekturer for å se om du kan vippe vektskålen i favør av dyp læring!  

---

💬 **Spørsmål eller tilbakemeldinger?**  
Ta gjerne kontakt eller opprett en issue—diskuterer alltid gjerne tidsserieprognoser! 🚀  

---