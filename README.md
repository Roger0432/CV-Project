# Sistema d'Anàlisi de Tràfic en Cruïlla

Aquest projecte implementa un sistema complet de visió per computador per a l'anàlisi de tràfic en cruïlles utilitzant models d'última generació.

## 🏗️ Arquitectura

- **Detecció**: YOLOv8 (detecta vehicles i vianants).
- **Seguiment (Tracking)**: ByteTrack (associa deteccions temporalment).
- **Assignació de Carrils**: Carrils virtuals definits per polígons i assignació basada en geometria.
- **Detecció d'Anomalies**:
    - Excés de velocitat.
    - Trajectòries inusuals (clustering).
    - Vianants a la calçada.

## 🚀 Instal·lació

1.  Clonar el repositori:
    ```bash
    git clone https://github.com/Roger0432/CV-Project.git
    cd CV-Project
    ```

2.  Crear un entorn virtual (opcional però recomanat):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```

3.  Instal·lar dependències:
    ```bash
    pip install -r requirements.txt
    ```

4.  Descarregar un vídeo del dataset UA-DETRAC (o utilitzar-ne un de propi) i guardar-lo a la carpeta `data/`.

## ⚙️ Configuració

Pots ajustar els paràmetres del sistema a `utils/config.py`:
- `VIDEO_PATH`: Ruta al vídeo d'entrada.
- `CAMERA_CALIBRATION_FACTOR`: Metres per píxel (calibrar segons la càmera).
- `LANE_POLYGONS`: Coordenades dels polígons dels carrils virtuals.
- `SPEED_THRESHOLD`: Límit per detectar excés de velocitat (km/h).

## ▶️ Execució

Per executar el pipeline complet d'anàlisi:

```bash
python src/main.py
```

## 📊 Resultats

Els resultats es guardaran a:
- `results/output_video.mp4`: Vídeo processat amb visualitzacions.
- `results/tracking_data.json`: Dades de trajectòries estructurades.
- `results/anomalies.csv`: Registre d'anomalies detectades.

## 🛠️ Estructura de Directoris

```
CV-Project/
├── data/           # Vídeos d'entrada
├── results/        # Sortides generades
├── src/            # Codi font dels mòduls
├── utils/          # Utilitats i configuració
├── requirements.txt
└── README.md
```
