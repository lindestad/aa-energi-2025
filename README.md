# aa-energi-2025

**Project information in English:** [README-ENGLISH.md](README-ENGLISH.md)


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