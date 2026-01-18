# Projekt: Przetwarzanie Języka Naturalnego
## Język analizowany: POLSKI

Projekt składa się z trzech niezależnych, ale logicznie powiązanych zadań:
1. Analiza statystyczna języka (korpus)
2. Przetwarzanie syntaktyczne (generator zdań SVO)
3. Analiza semantyczna (kolokacje i związki znaczeniowe)

Każde zadanie musi zostać zrealizowane jako działający program lub moduł.

---

## ZADANIE 1 – Statystyczna analiza języka naturalnego

### Cel
Empiryczne zbadanie własności statystycznych języka polskiego oraz weryfikacja prawa George’a Zipfa na podstawie dużego korpusu tekstów.

### Dane wejściowe
- Korpus tekstów w języku polskim
- Wielkość korpusu: około 100 000 wyrazów
- Teksty powinny pochodzić z różnych dziedzin (korpus zróżnicowany tematycznie)

### Wymagania funkcjonalne
Program musi:
1. Przeprowadzić tokenizację tekstu (podział na wyrazy).
2. Zliczyć częstość występowania każdego wyrazu.
3. Utworzyć ranking wyrazów według częstości.
4. Zweryfikować prawo Zipfa:
   - sprawdzić zależność:  
     `ranga wyrazu × liczba wystąpień ≈ const`
5. Wskazać:
   - wyrazy o najwyższej częstości (głównie wyrazy funkcyjne: przyimki, zaimki, partykuły),
   - wyrazy niosące główne znaczenie (środkowa część rankingu).
6. Obliczyć, ile wyrazów należy znać, aby rozumieć 90% korpusu.
7. Wygenerować listę wyrazów pokrywających 90% tekstu.
8. Zidentyfikować tzw. „rdzeń języka”:
   - wyrazy o bardzo dużej liczbie połączeń,
   - pełniące istotne funkcje składniowe.
9. (Opcjonalnie) Spróbować automatycznie przypisać znaczenie do ok. 50% rzeczowników na podstawie kontekstu.

### Wyniki
- Statystyki częstości wyrazów
- Ranking wyrazów
- Analiza i wizualizacja prawa Zipfa
- Lista słów pokrywających 90% korpusu

---

## ZADANIE 2 – Przetwarzanie syntaktyczne (generator zdań SVO)

### Cel
Zbudowanie generatora i korektora prostych zdań w języku polskim, opartych na strukturze SVO (Subject–Verb–Object), przeznaczonego dla osób uczących się języka.

### Forma
- Program z graficznym interfejsem użytkownika (GUI)
- Interfejs musi prowadzić użytkownika krok po kroku
- Program działa wyłącznie na zdaniach prostych

### Struktura zdania
Zdanie ma postać:

---

### Budowa podmiotu (S) i dopełnienia (O)

#### Typy
- Zaimek osobowy
- Fraza rzeczownikowa:
  - sam rzeczownik
  - rzeczownik + przymiotnik

#### Zasoby
- Lista 100 rzeczowników
- Lista 100 przymiotników

#### Cechy gramatyczne
Użytkownik wybiera:
- liczbę: pojedyncza / mnoga
- rodzaj:
  - męski
  - żeński
  - nijaki
  - męskoosobowy
  - niemęskoosobowy
- osobę (jeśli dotyczy)
- elementy dodatkowe:
  - określoność / nieokreśloność
  - zaimki dzierżawcze
  - zaimki wskazujące

Program musi:
- automatycznie dobrać poprawną formę fleksyjną
- uwzględnić przypadki gramatyczne języka polskiego

---

### Budowa orzeczenia (V)

#### Zasoby
- Lista 100 czasowników

#### Wybory użytkownika
- typ zdania:
  - twierdzące
  - przeczące
  - pytające
  - rozkazujące
  - przypuszczające
- czas gramatyczny
- czasownik w formie podstawowej

Program musi:
- automatycznie wygenerować poprawną formę pochodną czasownika
- dostosować formę do podmiotu (osoba, liczba, rodzaj)

---

### Funkcja korekty
Program musi:
- sprawdzić poprawność składniową zdania
- wskazać błędy gramatyczne
- zaproponować poprawną wersję zdania

---

## ZADANIE 3 – Analiza semantyczna i analiza kolokacji

### Cel
Analiza związków semantycznych między wyrazami na podstawie ich rzeczywistych połączeń w korpusie tekstów.

---

### Część A – Przymiotnik + rzeczownik

#### Dane
- 100 przymiotników
- 100 rzeczowników

#### Wymagania
1. Na podstawie korpusu zidentyfikować rzeczywiste połączenia przymiotnik–rzeczownik.
2. Uwzględnić różne formy fleksyjne (lemmatyzacja).
3. Zbudować graf dwudzielny:
   - wierzchołki: przymiotniki i rzeczowniki
   - krawędzie: wspólne wystąpienia w tekstach

---

### Część B – Czasownik przechodni + rzeczownik

#### Dane
- 100 czasowników przechodnich
- 100 rzeczowników (dopełnień)

#### Wymagania
1. Zidentyfikować połączenia typu czasownik–dopełnienie (np. „pić wodę”).
2. Zbudować analogiczny graf połączeń semantycznych.

---

### Ocena poprawności semantycznej

Program musi:
- analizować liczbę wystąpień danego połączenia w korpusie
- oceniać naturalność i poprawność sformułowania

#### Przykładowa skala
- 0 wystąpień → bardzo nienaturalne (czerwony)
- 1 wystąpienie → rzadkie (żółty)
- kilka wystąpień → poprawne (zielony)
- dużo wystąpień → bardzo naturalne (niebieski)

---

## Wymagania ogólne

- Wszystkie analizy dotyczą języka polskiego
- Należy uwzględnić:
  - fleksję
  - sprowadzanie wyrazów do formy podstawowej (lematyzacja)
  - analizę opartą na rzeczywistym korpusie tekstów
- Projekt może być:
  - jednym systemem z trzema modułami
  - lub trzema oddzielnymi aplikacjami
