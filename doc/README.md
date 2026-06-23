# Dokumentation Voronoi Projekt

- Modul: _Fachpraktikum Parallel Programming - 63782 - Prof. Dr. Lena Oden_
- Gruppe B:
  - _Pius Großmann_
  - _Anton Böhler_

---

# Vorgehen und Aufbau

Um ein strukturiertes Vorgehen zu gewährleisten wurde vorab ein Konzept entwickelt. Es beginnt mit dem ersten Abschnitt bei dem das Thema dieses Projekts klar beschrieben und abgesteckt wird.
Im darauf folgendem Abschnitt wird ein Performance Analyse Konzept entwickelt und implementiert und in allen anderen Abschnitten einheitliche Zeitmessungen und Analysen durchzuführen.
Der dritte Abschnitt befasst sich einer Naiven Implementation, wobei geklärt wird wie und warum diese funktioniert. Danach werden in Abschnitt vier und fünf die aus dem Kurs-Text erlernten Optimierungs-Verfahren angewandt. In Abschnitt sechs wird ein Algorithmus aus der Literatur erklärt, implementiert und mit dem bisherigem Algorithmus verglichen.
Zuletzt wird in Abschnitt sieben eine finale Analyse und Zusammenfassung der Ergebnisse dargelegt.

# Aufgabe 1 - Beschreibung des Problems

_Was ist das Problem?_

_2-3 wissenschaftliche Quellen_

_Was sind verwandte Probleme die nicht berücksichtigt werden?_

_Was wird berechnet?_

_Welche Einschränkungen beziehungsweise Annahmen werden gemacht?_

_Was ist die Eingabe und Ausgabe und welcher Daten-Typ wird genutzt?_

_Welche Parameter sind entscheidend für das Problem und welchen Einfluss haben diese?_

# Aufgabe 2 - Performance Analyse Konzept

```
TODO

- 512, 1024, ... 2^16
- log scale (x & y)
- single or multiple lines
- kernel only (no transfer or context switching)
```

_Wie werden im folgenden Performance Analysen durchgeführt?_

_Wie wird die Zeit für das kompilieren und den Daten-Transfer in der Analyse berücksichtigt?_

_Welche Eingabe- beziehungsweise Ausgabe-Größen werden verwendet?_

# Aufgabe 3 - Naive Implementation

_Wie viele Threads werden gestartet und welche Aufgabe hat ein jeder?_

Für jeden Pixel des Ergebnis wird ein Thread initialisiert. Jeder Thread iteriert durch alle Punkte und berechnet die Distanz zu jedem Punkt. Der Punkt mit der geringsten Distanz wird dabei gefunden und das Ergebnis in die Ausgabe geschrieben.

_Wie wird entschieden ob ein Punkt nächster Nachbar ist?_

Beim iterieren wird die Distanz zu jedem Punkt mit Hilfe von `cuda.libdevice.hypotf` berechnet und der Index des am nächsten liegenden Punkt gespeichert. Wenn nun ein Punkt mit geringerer Distanz gefunden wird, wird der Index überschrieben.

Folgende Animation gibt an, wie das Ergebnis nach jeder Iteration, also Hinzunahme eines weiteren Punkt, aussieht.

| Voronoi                                        | Distanzen                                            |
| ---------------------------------------------- | ---------------------------------------------------- |
| ![](../data/task3_euclidean_visualization.gif) | ![](../data/task3_euclidean_field_visualization.gif) |

Aus dieser Animation ist ersichtlich, dass jeder Pixel immer den bisherigen nächsten Nachbar verwaltet und inkrementell weitere Punkte hinzunimmt. In der beistehenden Animation sind die Distanzen zu sehen. Hierbei ist zu erkennen, dass die Distanzen sich mit jeder Iterationen verringern (oder gleich bleiben).

> [!note]
> Das berechneten Distanzen wurden aus dem Wertebereich $0$ bis $\sqrt{2}$ in den Wertebereich $0$ bis $255$ abgebildet. Zur besseren Visualisierung wurden die Distanzen mit dem Faktor $4$ hoch-skaliert und an der Oberen-Grenze abgeschnitten. Die Begründung hierfür ist, dass auch bei den letzten Iterationen der Animation noch Änderungen mit bloßem Auge zu erkennen sind. Für andere Animationen der Distanzen wird ebenfalls mit dem gleichen Faktor hoch-skaliert um Vergleiche zu ermöglichen.

_Wieso arbeitet der Algorithmus korrekt?_

Der Algorithmus arbeitet korrekt, weil für jeden Pixel jeder Punkt bei der Suche nach dem nächsten Nachbar berücksichtigt wird. Es kann also keinen Punkt geben der näher liegt als der berechnete Punkt.

_Müssen Race-Conditions beachtet werden?_

Da der Algorithmus für jeden Pixel einen Thread startet und Threads sich um nur deren zugewiesenen Pixel kümmern gibt es keine Race-Conditions. Es müssen also keine Atomic- oder Sync-Operationen durchgeführt werden.

_Gibt es warp divergence in dieser Implementation?_

Diese Implementation hat sehr wenig warp divergence. Bei dem Kernel Aufruf terminieren Threads die Pixel außerhalb des Ergebnis berechnen würden frühzeitig. Die Schleife wird in jedem Thread für jeden Punkt der Eingabe einmal ausgeführt. Das bedeutet jeder Thread führt die Schleife genau gleich-oft aus. An dieser Stelle gibt es also keine warp divergence. Hingegen bei der Verzweigung, ob die neu-berechnete Distanz näher liegt, kann es zu warp divergence kommen. Hierbei gibt es jedoch eine kleine Besonderheit. Die Pixel eines Warp liegen beieinander, weswegen Abstände zu den meisten Punkten ähnlich ausfallen und in vielen Fällen keine warp divergence auftritt.

_Welche Probleme beziehungsweise Grenzen hat der Kernel?_

Die Ausführungszeit eines Thread des Kernel wächst linear mit der Anzahl an Punkten. Gleiches gilt für die Anzahl an benötigten Threads in Bezug auf die Ausgabe-Größe. Diese Implementation kann keinen Punkt überspringen, da der Algorithmus dadurch nicht korrekt arbeiten würde.

_Welche Parameter haben den größten Einfluss auf die Performance und wieso?_

Die Anzahl an Punkten und die Ausgabe-Größe haben den größten Einfluss auf die Performance, da weitere Schleifen-Iterationen durchgeführt beziehungsweise weitere Threads gestartet werden müssen.

Folgendes Diagramm gibt die Laufzeit für eine feste Ausgabe-Größe von `128x128`.

| RTX-5070                                                                                                                           |     |
| ---------------------------------------------------------------------------------------------------------------------------------- | --- |
| ![](../data/performance_matrix_NVIDIA-GeForce-RTX-5070_euclidean_hypot_resolution=128,256,512,1024,2048_points=64,128,256,512.png) |     |
| ![](../data/performance_plot_NVIDIA-GeForce-RTX-5070_euclidean_hypot_resolution=128_points=64,128,256,512.png)                     |     |

Es ist leicht zu erkennen, dass größenordnungsmäßig ein verdoppeln der Anzahl an Punkten ein verdoppeln der Laufzeit mit sich bringt. Ein verdoppeln der Auflösung führt größenordnungsmäßig zu einem vervierfachen der Laufzeit. Das liegt daran, dass ein verdoppeln der Auflösung dazu führt, dass viermal so viele Pixel berechnet werden müssen.

Um eine nahe-liegende Optimierung zu motivieren, wird als Exkurs die Manhattan-Distanz betrachtet.

_Exkurs Manhattan-Distanz: Welchen Einfluss hat die Manhattan-Distanz als alternative Metrik, und wie verändert sich das visuelle Ergebnis?_

Die Manhattan-Distanz ist eine alternative Distanz-Funktion zur Euklidischen Distanz-Funktion. Sie ist definiert als die Summe der absoluten Abstände für jede Dimension.

$$\sum_{i=0}^n \left|a_i - b_i\right|$$

Der zuvor beschriebene Algorithmus ist Distanz-Funktion agnostisch, weswegen nur eine kleine Änderung nötig ist, um das Voronoi-Diagram mit Manhattan-Distanz-Funktion zu berechnen.

Folgende Animation wurde für die Manhattan-Distanz erstellt.

| Voronoi                                        | Distanzen                                            |
| ---------------------------------------------- | ---------------------------------------------------- |
| ![](../data/task3_manhattan_visualization.gif) | ![](../data/task3_manhattan_field_visualization.gif) |

Der Algorithmus funktioniert auf die gleiche Weise, es gibt jedoch abgesehen von der Ausgabe einen deutlichen Unterschied in der Laufzeit.
Der gleiche Algorithmus mit Manhattan-Distanz ist deutlich schneller als mit Euklidischer-Distanz, wie folgendes Diagramm zeigt.

| RTX-5070                                                                                                                 |     |
| ------------------------------------------------------------------------------------------------------------------------ | --- |
| ![](../data/performance_plot_NVIDIA-GeForce-RTX-5070_euclidean_hypot_manhattan_resolution=128_points=64,128,256,512.png) |     |

Dies ist natürlich nicht verwunderlich, da für die Berechnung der Manhattan-Distanz nur Addition und die Absolut-Funktion nötig sind, welche sehr leicht zu berechnen sind.
Hingegen für die Euklidische-Distanz wird die `cuda.libdevice.hypotf` Funktion aufgerufen, welche schwerer beziehungsweise langsamer zu berechnen ist.
Es ist also ersichtlich, dass durch eine effizientere Distanz-Prüfung eine schnellere Laufzeit möglich ist.

# Aufgabe 4 - Optimierung durch billigere Distanz-Prüfung

_Welche Optimierungen können bei der Distanz-Prüfung problemlos gemacht werden und warum?_

Der bisherige Algorithmus arbeitet jeden Punkt ab und berechnet den Abstand um einen nächsten Nachbar zu bestimmen. Es gibt zwei Ansätze die an dieser Stelle untersucht werden können.

1. Schnellere Distanz Berechnung

Es kann versucht werden das berechnen von Distanz-Funktionen zu verbessern. Hierbei gibt es die Möglichkeit die `sqrt` Funktion aufzurufen.

Ansonsten können schnellere Mathe-Operationen ermöglicht werden mit der `fastmath=True` Annotation. Laut dem [nvidia-numba-cuda-user-guide](https://nvidia.github.io/numba-cuda/user/fastmath.html) werden einige Operationen wie `sqrt` durch schnellere Approximationen ersetzt und Multiplikations- und Additions-Operationen verschmolzen. Da unser Algorithmus nicht die exakten Distanzen benötigt sondern nur Distanzen vergleichen muss ist dies ein klarer Anwendungsfall.

Zuletzt kann darauf verzichtet werden die tatsächliche Euklidische Distanz zu berechnen. Da nur Distanzen verglichen werden, können wir auch die quadratische euklidische Distanz vergleichen. Auf den ersten Blick erscheint dies aufwendiger, jedoch kann auf diese Weise auf alle kompliziertere Mathe-Operationen verzichtet werden.

2. Distanz Berechnung überspringen

Es können schnelle Approximationen der Distanz-Funktion verwendet werden um sich die exakte Berechnung einzusparen. Beispielsweise kann der Abstand unter Berücksichtigung nur einer Dimension bestimmt werden. Auf diese Weise können Punkte die definitiv zu Weit entfernt sind übersprungen werden, indem eine Verzweigung eingesetzt wird.

_Welchen Einfluss hat `sqrt` und wieso?_

Für die bisherigen Berechnungen wurde die `cuda.libdevice.hypotf` Funktion verwendet, da diese genau die Euklidische-Distanz berechnet. Es ist auch möglich das equivalent der `cuda.libdevice.hypotf` Funktion zu berechnen, indem der Ausdruck $\sqrt{(a_x - b_x)^2 + (a_y - b_y)^2}$ ausgeschrieben wird mit Hilfe der Funktion `cuda.libdevice.sqrtf`.

Die Annahme vorab ist, dass die beiden Funktionen die gleiche Laufzeit haben. Es wäre auch denkbar, dass die `cuda.libdevice.hypotf` Funktion bestimmte Optimierungen ermöglicht, die bei der generischen Funktion `cuda.libdevice.sqrtf` nicht möglich wären. Tatsächlich hat sich in der Praxis das gegenteil gezeigt, wie folgender Ausschnitt aus dem Assembly zeigt.

```diff
$L__BB0_6:
	mul.lo.s64 	%rd66, %rd76, %rd8;
	add.s64 	%rd67, %rd1, %rd66;
	add.s64 	%rd68, %rd5, %rd66;
	ld.global.b32 	%r132, [%rd67];
	add.s64 	%rd69, %rd9, %rd68;
-	ld.b32 	%r133, [%rd69];
+   ld.b32 	%r61, [%rd69];
-	sub.f32 	%r134, %r1, %r132;
-	sub.f32 	%r135, %r2, %r133;
+	sub.f32 	%r62, %r1, %r60;
+	sub.f32 	%r63, %r2, %r61;
-	abs.f32 	%r136, %r134;
-	abs.f32 	%r137, %r135;
-	min.s32 	%r138, %r137, %r136;
-	max.s32 	%r139, %r136, %r137;
-	and.b32 	%r140, %r139, -33554432;
-	xor.b32 	%r141, %r140, 2122317824;
-	mul.f32 	%r142, %r138, %r141;
-	mul.f32 	%r143, %r139, %r141;
-	mul.f32 	%r144, %r142, %r142;
-	fma.rn.f32 	%r145, %r143, %r143, %r144;
-	sqrt.rn.f32 	%r146, %r145;
+	mul.f32 	%r64, %r63, %r63;
+	fma.rn.f32 	%r65, %r62, %r62, %r64;
+	sqrt.rn.f32 	%r66, %r65;
-	or.b32 	%r147, %r140, 8388608;
-	mul.f32 	%r148, %r146, %r147;
-	setp.eq.f32 	%p36, %r138, 0f00000000;
-	selp.f32 	%r149, %r139, %r148, %p36;
-	setp.eq.f32 	%p37, %r138, 0f7F800000;
-	selp.f32 	%r150, 0f7F800000, %r149, %p37;
-	setp.lt.f32 	%p38, %r150, %r151;
-	selp.b64 	%rd75, %rd76, %rd75, %p38;
+	setp.lt.f32 	%p24, %r66, %r67;
+	selp.b64 	%rd75, %rd76, %rd75, %p24;
```

Es ist zu erkennen, dass die `cuda.libdevice.hypotf` zu einem größeren Ausdruck übersetzt wird und innerhalb dieses Ausdruck wird die intrinsic Funktion `sqrt.rn.f32` aufgerufen. Das explizite Ausschreiben des Ausdruck $\sqrt{(a_x - b_x)^2 + (a_y - b_y)^2}$ führt dazu, dass gewisse Anweisungen wegfallen.

Ein Abgleich mit der Dokumentation von CUDA für die [hypotf Funktion](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#group__cuda__math__single_1ga2880a4ebf5500aeb74fb01340ea91215) gibt einen Einblick weswegen diese Anweisungen existieren.
Die `cuda.libdevice.hypotf` Funktion muss garantieren, dass:
- $\mathrm{hypot}(x,y)$, $\mathrm{hypot}(y,x)$ und $\mathrm{hypot}(x,-y)$ äquivalent sind
- $\mathrm{hypot}(x, \pm 0)$ äquivalent zu $\mathrm{fabsf}(x)$ ist
- $\mathrm{hypot}(\pm \infty,y)$ immer $+ \infty$ ergibt, selbst wenn $y=\mathrm{NaN}$
- $\mathrm{hypot}(\mathrm{NaN},y)$ immer $\mathrm{NaN}$ ergibt, selbst wenn $y \neq \pm \infty$

Diese Bedingungen erfordern zusätzliche Anweisungen.

Für den Algorithmus sind die meisten dieser Bedingungen tatsächlich irrelevant. Das Verhalten des Algorithmus ändert sich nicht, wenn die Funktion $\mathrm{NaN}$ statt $\infty$ oder andersherum zurückgibt, da Distanzen nur verglichen werden und der Algorithmus in beiden Fällen die gleiche Verzweigung wählt.

Durch die verringerte Anzahl an Anweisungen ist eine klare Verbesserung in der Laufzeit zu erkennen.

| RTX-5070 |     |
| -------- | --- |
|  ![](../data/performance_matrix_NVIDIA-GeForce-RTX-5070_euclidean_sqrt_resolution=128,256,512,1024,2048_points=64,128,256,512.png)        |     |
|   ![](../data/performance_plot_NVIDIA-GeForce-RTX-5070_euclidean_hypot_euclidean_sqrt_resolution=128_points=64,128,256,512.png)       |     |

Es ist hierbei zu beachten, dass effektiv keine günstigere Distanz-Rechnung durchgeführt wurde, sondern es wurde auf gewisse Garantien bei der bisherigen Distanz-Berechnung verzichtet, da diese für den Algorithmus keinen Unterschied machen. Im folgenden wird auf exakte Ergebnisse verzichtet um den Algorithmus noch schneller zu machen.

_Welchen Einfluss hat `fastmath` und wieso?_

Wie bereits erwähnt ermöglicht die Annotation `fastmath=True`, auf Genauigkeit zu verzichten im Austausch für schnellere Laufzeiten. Als Beispiel ist folgender Ausschnitt aus dem Assembly der regulären `sqrtf`-Variante und der `sqrtf`-Variante mit `fastmath=True` gegeben.

```diff
$L__BB0_6:
	mul.lo.s64 	%rd66, %rd76, %rd8;
	add.s64 	%rd67, %rd1, %rd66;
	add.s64 	%rd68, %rd5, %rd66;
	ld.global.b32 	%r60, [%rd67];
	add.s64 	%rd69, %rd9, %rd68;
	ld.b32 	%r61, [%rd69];
	sub.f32 	%r62, %r1, %r60;
	sub.f32 	%r63, %r2, %r61;
	mul.f32 	%r64, %r63, %r63;
	fma.rn.f32 	%r65, %r62, %r62, %r64;
-	sqrt.rn.f32 	%r66, %r65;
+	sqrt.approx.ftz.f32 	%r66, %r65;
	setp.lt.f32 	%p24, %r66, %r67;
	selp.b64 	%rd75, %rd76, %rd75, %p24;
```

In diesem Fall hat der Compiler wegen `fastmath=True` die intrinsic Funktion `sqrt.approx.ftz.f32` statt `sqrt.rn.f32` ausgewählt. Gleichermaßen wurden Divisionen `div.rn.f32` durch `div.approx.ftz.f32` ersetzt. Abgesehen von diesen beiden Operationen bleibt das Assembly größtenteils gleich.

Für die Laufzeit ergeben sich folgende Unterschiede.

| RTX-5070 |     |
| -------- | --- |
|  ![](../data/performance_matrix_NVIDIA-GeForce-RTX-5070_euclidean_sqrt_fast_resolution=128,256,512,1024,2048_points=64,128,256,512.png)        |     |
|   ![](../data/performance_plot_NVIDIA-GeForce-RTX-5070_euclidean_sqrt_euclidean_sqrt_fast_resolution=128_points=64,128,256,512.png)       |     |



_Welchen Einfluss hat schlicht $a^2 + b^2$ und wieso?_

# Aufgabe 5 - Effizienteres Laden von Daten

_Wie wurden Daten in der Naiven Implementation geladen?_

Die Naive Implementation iteriert in jedem Thread mit einer Schleife über die Punkte. Das bedeutet jeder Thread greift in den globalen Speicher für jeden Punkt.

_Was kann beim Laden der Daten verbessert werden?_

Da in der Naiven Implementation zu jedem Zeitpunkt bekannt ist, welcher Punkt als nächstes benötigt wird, wäre es möglich die Daten in mehreren Bündeln (Batch-Processing) zu verarbeiten. Das bedeutet, dass ein Teil der Daten gleichzeitig geladen wird und danach gleichzeitig verarbeitet wird.

_Wie können die Daten ins Shared Memory geladen und mit einem Grid-Stride-Loop verarbeitet werden?_

Der gewählte Ansatz besteht darin eine feste Konstante `GRID_STRIDE_SIZE` zu definieren. Auf diese weise kann ein `cuda.shared.array` definiert und im Kernel zugegriffen werden. Dieser hat wie die Punkte Eingabe auch zwei Dimensionen, nämlich `GRID_STRIDE_SIZE` und `2`. Nun werden die ersten `GRID_STRIDE_SIZE` Punkte in das Shared Memory geschrieben. Hierbei werden `2 * GRID_STRIDE_SIZE` Threads benötigt, da für jeden Punkt ein `x` und ein `y` geladen werden muss. Je zwei aufeinander folgende Threads laden also die Daten für einen Punkt. Threads die keinen Punkt berechnen sollen warten beziehungsweise falls keine Punkte mehr vorhanden sind wird `np.inf` ins Shared Memory geschrieben um Fehler bei Rechnungen zu vermeiden. Bevor lesend auf das Shared Memory zugegriffen werden kann, muss `cuda.syncthreads()` aufgerufen werden, um Race-Conditions zu vermeiden. Nun können alle Threads aus dem Block durch das Array iterieren und Distanzen berechnen. Falls ein neuer Nächster Nachbar entdeckt wurde, muss der korrekte index des Punkt berechnet werden (Der Index in das Shared Memory Array wäre nicht korrekt). Zuletzt muss erneut `cuda.syncthreads()` aufgerufen werden, bevor die Schleife sich wiederholt, erneut um Race-Conditions zu vermeiden.

Ein weiteres Detail ist, dass ein early-exit nicht mehr möglich ist für Threads die Pixel außerhalb des Diagram berechnen. Das liegt daran, dass Threads neben dem Pixel ausrechnen auch Punkte laden müssen. Es kann also sein, dass ein Thread zwar außerhalb des Diagram liegt, aber trotzdem Punkte für andere Threads laden muss. Erst nachdem keine Punkte mehr geladen werden müssen ist ein exit für diese Threads möglich beziehungsweise nötig.

_Welchen Einfluss hat das Laden der Daten ins Shared Memory und Verarbeiten mit einem Grid-Stride-Loop und wieso?_

_Wie können die Daten innerhalb eines Warp geladen und mit `shfl_sync` verarbeitet werden werden?_

Das Verfahren hat starke Ähnlichkeit mit dem vorherigen Ansatz. In diesem Fall werden Daten nicht mehr auf auf Block-Ebene geladen, sondern auf Warp-Ebene. Da Threads in einem Warp synchron ablaufen ist kein Aufruf von `cuda.syncthreads()` mehr nötig. Ein Nachteil hierbei ist, dass nun jeder Warp die Daten laden muss. Da nun kein `cuda.shared.array` vorhanden ist, muss eine lokale Variable definiert werden, welches auf ähnliche Weise verwendet wird. Die Variable wurde `point_component_warp_value` benannt und wird für jeden zweiten Thread eines Warp die `x`-Komponente und für jeden anderen Thread des Warp die `x`-Komponente laden. Falls ein Punkt nicht vorhanden ist, werden die Komponenten jeweils auf `np.inf` gesetzt um Fehler bei Rechnungen zu vermeiden. Wenn ein Warp nun die Variable `point_component_warp_value` eines jeden Thread befüllt wurden 16 `x`- und 16 `y`-Komponenten geladen, da es insgesamt 32 Threads pro Warp sind. Nun kann mit `cuda.shfl_sync` auf einen beliebigen Wert eines anderen Thread des gleichen Warp zugegriffen werden. Mit `cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index)` und `cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index + 1)` werden die Komponenten eines Punkt geladen und es kann die Distanz-Rechnung durchgeführt werden. Die beiden `cuda.shfl_sync` Aufrufe werden 16-mal wiederholt, bis vom Warp geladenen alle Punkte verarbeitet sind. Danach können die nächsten Punkte geladen werden, erneut ohne ein Aufruf von `cuda.syncthreads()`, da die Threads eines Warp synchron ablaufen.

Wie auch beim vorherigen Ansatz ist ein early-exit nicht möglich aus den gleichen Gründen.

_Welchen Einfluss hat das Laden der Daten in einen Warp und das Verarbeiten mit `shfl_sync` und wieso?_

# Aufgabe 6 - Alternativer Ansatz: Jump Flooding Algorithmus (JFA)

_Wie funktioniert der Algorithmus?_

_Wo liegen die Unterschiede zum Ansatz der vorherigen Implementierung? (Komplexität)_

_Können Optimierungen durchgeführt werden? wieso Ja/ Nein?_

_Gibt es Qualitätsunterschiede (Pixelfehler) im Diagramm?_

Die folgenden _Error Maps_ zeigen die Abweichungen der verschiedenen JFA-Varianten im Vergleich zum Pixel-Algorithmus, welcher hier als exakte `100%`-Referenz dient (schwarz = identisch, rot = Abweichung).

| `Standard JFA`                                | `JFA+1`                                     | `JFA+2`                                     |
| --------------------------------------------- | ------------------------------------------- | ------------------------------------------- |
| ![](../data/task6_error_map_standard_jfa.jpg) | ![](../data/task6_error_map_jfa_plus_1.jpg) | ![](../data/task6_error_map_jfa_plus_2.jpg) |

```bash
Standard JFA: 99.5950%
JFA + 1: 99.5990%
JFA + 2: 99.5993%
```

# Aufgabe 7 - Ergebnisse

_Welche der Optimierungen hat den größten Laufzeit-gewinn erbracht?_

_Wie viel schneller ist die Finale Implementation im Vergleich zur Naiven Implementation?_

_Welchen Durchsatz haben die verschiedenen Implementationen?_

_Ab welcher Eingabe-Größe erreicht die GPU ihre Sättigung?_

_Was liefern die Profiling-Tools?_

# Anhang

> [!NOTE]
> **Einschränkung bei Keyword (Named)-Arguments in CUDA-Kernels**
>
> Je nach installierter **Numba**-Version kann es zu Problemen kommen, wenn innerhalb eines CUDA-Kernels Keyword-Arguments (benannte Argumente) für Device-Funktionen verwendet werden.
>
> - **Problem:** Die Kompilierung schlägt fehl und scheint direkt bei der Kernel-Signatur "stehenzubleiben". Der Traceback zeigt jedoch, dass die Ursache weiter unten im Kernel liegt. Ein Aufruf der Art `get_thread_position(image=out_image)` kann nicht verarbeitet werden
> - **Lösung:** Den Aufruf auf positionale Argumente umstellen: `get_thread_position(out_image)`
>
> Verwendete Versionen anzeigen: `pip freeze`
