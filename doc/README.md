# Dokumentation Voronoi Projekt

- Modul: _Fachpraktikum Parallel Programming - 63782 - Prof. Dr. Lena Oden_
- Gruppe B:
  - _Pius Großmann_
  - _Anton Böhler_

---

# Aufgabe 1 - Beschreibung des Problems

_Was ist das Problem?_

_2–3 wissenschaftliche Quellen_

_Was sind verwandte Probleme die nicht berücksichtigt werden?_

_Was wird berechnet?_

_Welche Einschränkungen beziehungsweise Annahmen werden gemacht?_

_Was ist die Eingabe und Ausgabe und welcher Daten-Typ wird genutzt?_

_Welche Parameter sind entscheidend für das Problem und welchen Einfluss haben diese?_

# Aufgabe 2 - Performance Analyse Konzept

_Wie werden im folgenden Performance Analysen durchgeführt?_

_Wie wird die Zeit für das kompilieren und den Daten-Transfer in der Analyse berücksichtigt?_

_Welche Eingabe- beziehungsweise Ausgabe-Größen werden verwendet?_

# Aufgabe 3 - Naive Implementation

_Wie viele Threads werden gestartet und welche Aufgabe hat ein jeder?_

Für jeden Pixel des Ergebnis wird ein Thread initialisiert. Jeder Thread iteriert durch alle Punkte und berechnet die Distanz zu jedem Punkt. Der Punkt mit der geringsten Distanz wird dabei gefunden und das Ergebnis in die Ausgabe geschrieben.

_Wie wird entschieden ob ein Punkt nächster Nachbar ist?_

_Wieso arbeitet der Algorithmus korrekt?_

_Müssen Race-Conditions beachtet werden?_

_Gibt es warp divergence in dieser Implementation?_

_Welche Probleme beziehungsweise Grenzen hat der Kernel?_

_Welche Parameter haben den größten Einfluss auf die Performance und wieso?_

# Aufgabe 4 - Optimierung durch billigere Distanz-Prüfung

_Welche Optimierungen können bei der Distanz-Prüfung problemlos gemacht werden und warum?_

_Welchen Einfluss hat `fastmath` und wieso?_

_Welchen Einfluss hat `hypot` und wieso?_

_Welchen Einfluss hat schlicht $a^2 + b^2$ und wieso?_

_Welchen Einfluss hat ein approximiertes `if` vor der Distanz-Prüfung und wieso?_

_Welchen Einfluss hat die **Manhattan-Distanz** als alternative Metrik, und wie verändert sich das visuelle Ergebnis?_

# Aufgabe 5 - Effizienteres Laden von Daten

_Wie wurden Daten in der Naiven implementation geladen?_

_Was kann beim Laden der Daten verbessert werden?_

_Welchen Einfluss hat das Laden der Daten ins Shared Memory und Verarbeiten mit einem Grid-Stride-Loop wieso?_

_Welchen Einfluss hat das Laden der Daten in einen Warp und das Verarbeiten mit `shfl_down_sync` und wieso?_

_Welchen Einfluss hat ein Loop-Unrolling_

# Aufgabe 6 - Alternativer Ansatz: Jump Flooding Algorithmus (JFA)

_Wie funktioniert der Algorithmus?_

_Wo liegen die Unterschiede zum Ansatz der vorherigen Implementierung? (Komplexität)_

_Können Optimierungen durchgeführt werden? wieso Ja/ Nein?_

_Gibt es Qualitätsunterschiede (Pixelfehler) im Diagramm?_

# Aufgabe 7 - Ergebnisse

_Welche der Optimierungen hat den größten Laufzeit-gewinn erbracht?_

_Wie viel schneller ist die Finale Implementation im Vergleich zur Naiven Implementation?_

_Welchen Durchsatz haben die verschiedenen Implementationen?_

_Ab welcher Eingabe-Größe erreicht die GPU ihre Sättigung?_

_Was liefern die Profiling-Tools?_
