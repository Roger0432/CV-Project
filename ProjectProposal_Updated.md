# Project Proposal Updates

## 3.2.3 Speed Estimation

We propose a single robust method for speed estimation using homography transformation (Plan A), discarding the relative speed approach due to its dependency on unknown variables.

**Methodology:**
1.  **Scene Calibration:** We compute a homography matrix using at least 4 point correspondences between the image plane and the road plane. These points are derived from road markings with known standard real-world dimensions (e.g., lane line length, width, or distance between dash markings).
2.  **Speed Calculation:** For each tracked vehicle, we map the pixel centroid coordinates $(u, v)$ to real-world coordinates $(X, Y)$ using the homography matrix. The instantaneous speed is calculated as the Euclidean distance between consecutive real-world positions divided by the time elapsed (derived from the video frame rate).
    $$ Speed = \frac{\sqrt{(X_t - X_{t-1})^2 + (Y_t - Y_{t-1})^2}}{\Delta t} $$
3.  **Smoothing:** To reduce noise from detection jitter, we apply a sliding window average over the computed instantaneous speeds.

**Reference Speed (Ground Truth):**
To respond to the question "What will be the reference speed?", we generate a high-quality Ground Truth dataset manually.
-   **Procedure:** We identify a road segment of known length (e.g., a 10m pedestrian crossing or marked lane section).
-   **Annotation:** We manually annotate the entry and exit timestamps for 20-30 distinct vehicles passing through this segment.
-   **Calculation:** $Reference Speed = \frac{Known Distance}{Exit Time - Entry Time}$.

**Fast vs. Normal Cars:**
We define "Fast" cars based on both relative statistics and absolute safety limits:
-   **Definition:** A vehicle is classified as "Fast" if its estimated speed is $> 1.3 \times$ the average lane speed OR if it exceeds an absolute threshold of $50 \text{ km/h}$.
-   **Evaluation:** We evaluate this classification on the same 20-30 labeled vehicles using Precision, Recall, and F1-score metrics for the "Fast" class.

**Performance Metrics:**
-   **MAE (Mean Absolute Error):** We quantify the accuracy of our speed estimation by computing the MAE between our estimated speed (using homography) and the manually calculated Reference Speed for the annotated set.
    $$ MAE = \frac{1}{N} \sum_{i=1}^{N} |EstimatedSpeed_i - ReferenceSpeed_i| $$

## 3.2.4 Trajectory Anomaly Detection

We focus on detecting three specific types of traffic violations using trajectory analysis.

**Dataset Specification (Target):**
To ensure a concrete evaluation protocol, we **aim to construct** our test set as follows:
-   **Volume:** 10 video clips (10-20 seconds each).
-   **Total Trajectories:** Approximately 150 total vehicle trajectories.
-   **Anomalies:** The dataset **should contain approximately** **35 anomalous trajectories**, distributed as (approx. 3-4 per clip).

**Anomaly Definitions & Detection Logic:**
1.  **Wrong Direction (15 cases):**
    -   *Constraint:* Trajectory vector angle opposes the dominant lane flow direction ($> 150^\circ$ difference).
    -   *Method:* DBSCAN or K-means clustering on motion vectors to establish dominant flow.
2.  **Forbidden Zone (10 cases):**
    -   *Constraint:* Vehicle centroid enters a predefined polygon covering a pedestrian-only area or safety island.
    -   *Method:* Point-in-polygon geometric test.
3.  **Overspeeding (10 cases):**
    -   *Constraint:* Vehicle speed exceeds the "Fast" car threshold defined in Section 3.2.3.
    -   *Method:* Z-score analysis ($z > 3$) relative to lane distribution or absolute threshold check ($> 50 \text{ km/h}$).

## Summary Table: Planned Evaluation Metrics

| Task | Metric | Definition / Ground Truth |
| :--- | :--- | :--- |
| **Lane Assignment** | Accuracy | % of vehicles assigned to correct ID (Manual visual check) |
| **Speed Estimation** | MAE (km/h) | Difference vs. Manual GT (Dist/Time on 10m segment) |
| **Fast Car Det.** | F1-Score | Precision/Recall on "Fast" class (GT: 20-30 vehicles) |
| **Anomaly Det.** | Precision/Recall | Detection of 35 specific anomalies in 10 clips (150 trajectories) |
