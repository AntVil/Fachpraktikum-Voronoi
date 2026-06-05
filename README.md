# Fachpraktikum Bild-Filter

## Program Ausführen

- Eine CUDA-fähige GPU ist nötig für Ausführungen der Programme.

- Stelle sicher, dass der Befehl `nvidia-smi` funktioniert und eine sinnvolle Ausgabe gibt. Ansonsten ist eine Installation eines kompatiblen Treibers nötig.

```bash
nvidia-smi
```

### ... auf Windows

Dieser Guide nutzt **uv** als Python-Manager. Wer klassisches Python nutzt, ersetzt `uv pip` durch `pip` und `uv venv` durch `python -m venv`.

- **Vorbereitung (für uv-Nutzer):**

  Prüfe mit `uv python list`, ob bereits eine passende Version installiert ist. Wenn noch keine Python-Version installiert ist, kann für das Projekt beispielsweise Python 3.12 verwendet werden:

  ```powershell
  uv python install 3.12
  ```

- **Virtuelle Umgebung (venv) erstellen:**

  ```powershell
  uv venv --python 3.12
  ```

  Ersetze _3.12_ durch die passende Version.

- **Umgebung aktivieren:**

  ```powershell
  .venv\Scripts\activate
  ```

- **Abhängigkeiten installieren:**

  ```powershell
  uv pip install -r requirements.txt
  ```

- **Installation von numba-cuda:**

  Da `numba` aktuell nicht in der `requirements.txt` enthalten ist (um Hardware-Konflikte zu vermeiden), muss es separat installiert werden:
  1. **Manueller Weg (Mein Setup):** Ich habe das [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) systemweit installiert und die Pfade für `nvvm` und `cudart` manuell in den Umgebungsvariablen hinterlegt. In diesem Fall reicht die Installation des Basis-Pakets aus:

     ```powershell
     uv pip install numba
     ```

  2. **Automatischer Weg (Alternative):** Falls kein Toolkit installiert werden soll, können die benötigten Bibliotheken laut [NVIDIA docs](https://nvidia.github.io/numba-cuda/user/installation.html#installation-with-a-python-package-manager) direkt in die Umgebung geladen werden (passend zur Hardware/Treiber-Version):
     ```powershell
     uv pip install "numba-cuda[cu12]"  # oder [cu13]
     ```

  _Wichtig: Ein aktueller NVIDIA-Treiber (Check via `nvidia-smi`) muss in jedem Fall installiert sein._

- **Programm ausführen:**
  ```powershell
  uv run .\src\task1.py
  ```

### ... auf Linux

- Erstellen eines virtual environment:

```bash
python3 -m venv venv
```

- Aktivieren von environment:

```bash
source venv/bin/activate
```

- Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

- Zur korrekten installation von `numba-cuda` muss die CUDA-version ermittelt werden (zum Beispiel über `nvidia-smi`)
- Für `CUDA 13` wäre der Befehl (siehe https://nvidia.github.io/numba-cuda/user/installation.html):

```bash
pip install numba-cuda[cu13]
```

- Program ausführen:

```bash
python src/task1.py
```

### Fehlerbehebung

#### Installation Verifizieren

- Zur Überprüfung, ob `numba` auf die GPU zugreifen kann, kann folgender Befehl verwendet werden:

```bash
python -c "from numba import cuda;cuda.detect()"
```

- Falls beim Ausführen ein Fehler wie `No such file: libcudart.so*` oder `No such file: libnvvm.so*` auftritt ist es potentiell nötig die _CUDA Runtime API library_ zu installieren:

```bash
apt install nvidia-cuda-toolkit
```

## Dokumentation

Die Dokumentation ist im Ordner `doc/README.md` enthalten.
Darüber hinaus sind Code-Kommentare verfügbar.

## Nützliche Links

- [Markdown Syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Numba Documentation](https://numba.readthedocs.io/en/stable/)
- [Numba Cuda Documentation](https://nvidia.github.io/numba-cuda/user/index.html#user-guide)
