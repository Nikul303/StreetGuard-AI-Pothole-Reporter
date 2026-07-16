# StreetGuard-AI-Pothole-Reporter
"Technical review and documentation for AI-based pothole detection system"
 GitHub README and profile for



StreetGuard — AI Pothole Reporter
Abstract
StreetGuard is an AI-driven system for automatic identification, localization and reporting of road potholes using images/video captured by citizen smartphones or vehicle-mounted cameras. The system uses computer-vision models to detect and classify potholes, estimates severity and GPS location, stores evidence in a central database, and generates actionable reports (CSV / dashboard / notifications) for municipalities. This document presents the system architecture, data flow (input→output), database configuration, model & pipeline choices, evaluation metrics and expected end results.

Introduction and Scope
Objective: Provide a robust, deployable architecture for pothole detection & reporting without implementing code now — a technical design and documentation to guide future implementation.

Scope: Theory and system-level technical details including:

ML approach for detection/segmentation and severity estimation.
End-to-end architecture (mobile/edge capture → cloud/server → dashboard).
Input/output formats & API examples.
Database schema and storage design.
Expected outputs and evaluation metrics.
Reference: The structural approach takes inspiration from Gesture-Classification-System (https://github.com/Yashsmakwana/Gesture-Classification-System) — similar pipeline stages (dataset → preprocessing → model training → inference → reporting/UI) but adapted to pothole detection.

Need and Use-cases
Public safety: Quick reporting reduces vehicle damage and accidents.
Civic maintenance: Prioritize repairs using severity and location heatmaps.
Crowdsourced monitoring: Citizens and city vehicles feed continuous data.
Insurance claims: Visual evidence & timestamps help adjudication.
Technologies / Techniques Recommended
Computer Vision: Object detection (YOLO-family) and/or semantic segmentation (UNet/DeepLab) for accurate localization.
Backbone models: MobileNetV3, EfficientNet-lite or YOLOv5/YOLOv8 for mobile/edge-friendly inference.
Transfer learning: Pretrained COCO weights, fine-tuned on pothole dataset.
Data storage: PostgreSQL + PostGIS (for spatial queries), object store (Amazon S3 / MinIO) for images.
Model serving: TorchServe / TensorFlow Serving / ONNX Runtime or lightweight edge runtime (TensorRT/TF Lite).
Backend/API: Python (FastAPI / Flask), containerized (Docker), deployed on Kubernetes or cloud functions.
Frontend & Dashboard: React / Vue, map visualization (Leaflet, Mapbox, or Google Maps).
Geolocation: Use device GPS and reverse-geocoding for address context.
Message/Notifications: Email/SMS/webhooks for municipality integration.
Science / Concept Behind It
Detection: Object detection outputs bounding boxes + confidence for potholes.
Segmentation: Pixel-level mask gives precise area to estimate size and volume (approximate).
Severity Estimation: Derived from mask area in pixels + approximate scale (using known camera height or road marker) or using monocular depth estimation network for more accurate depth/volume.
Localization: GPS from device metadata; for video/vehicle-mounted cameras without accurate GPS, use simultaneous odometry/GNSS or successive frame triangulation.
Robustness: Data augmentation (brightness, blur, occlusion), handling different lighting and weather.
Methodology / Pipeline Overview
Stages:

Data collection

Images & videos with GPS/time metadata and labels (bbox/mask + severity).
Data preprocessing & augmentation

Resize, normalization, contrast/illumination augmentation, geotag extraction.
Model training

Train detection/segmentation model with transfer learning, use cross-validation.
Model validation & metrics

mAP for detection, IoU for segmentation, precision/recall, F1, severity regression error.
Deployment

Convert model to appropriate runtime (TorchScript/ONNX).
Inference & backend processing

Client sends image + metadata → server runs model → detects pothole(s) → computes severity + location → stores in DB → triggers report/update dashboard.
Human-in-the-loop verification (optional)

Allow municipal staff to review & confirm before physical dispatch.
System Architecture (Block Diagram + Components)

Logical components:

Client(s):
Mobile App (Android/iOS) or Edge Device (dashcam) capturing images/video and GPS/time.
Inference Endpoint:
Model Server (REST/gRPC inference) using torchserve/TF-Serving/ONNX Runtime.
Backend/API:
FastAPI application handling uploads, orchestration, auth, DB access, and report generation.
Storage:
Object store for images/videos, Postgres+PostGIS for structured storage, Redis for queue/caching.
Dashboard:
Web-based admin dashboard presenting map, report lists, analytics.
Notification Service:
Email/SMS or webhook integration.
Admin Tools:
Quality control, manual labeling interface (to retrain models), status tracking.
ASCII block diagram (simple):

[Mobile App / Vehicle Camera] | HTTP/HTTPS POST (image + metadata) | [API Gateway / Load Balancer] | [Backend / FastAPI] -----> [Object Storage (S3/MinIO)] (store image) | | | V |----------------> [Inference Service (model server)] (runs detection/segmentation) receives image URL, returns detections | Post-processing (severity calc, duplicate suppression) | [Postgres + PostGIS] <--- store report, detection metadata | [Dashboard / Notifications / Export]

Input → Output Execution Flow (Sequence)
Capture: User captures image/video via mobile app; app collects:
image file (jpeg/png), timestamp
GPS coordinates (lat, lon), accuracy
optional: device height, heading, user notes
Upload: App POSTs multipart/form-data to /api/reports/upload:
payload: image file; metadata JSON
API:
Validate auth & metadata.
Store original image to object storage (returns URL).
Push processing job to queue / call inference directly.
Inference:
Model receives image URL or bytes.
Runs detection/segmentation; returns:
detections: [{"bbox":[x,y,w,h], "mask":<RLE or mask URL>, "class":"pothole", "confidence":0.92, "severity_score":0.7}, ...]
processing_time_ms
Post-processing:
Convert bbox/mask size to estimated area (pixels → meters² using camera params or reference).
Severity categorization (e.g., minor/moderate/severe) using threshold rules or small regression model.
If multiple overlapping detections, merge.
Storage:
Insert report record in database
Save per-detection records (with image URL, bbox, mask URL)
Response to user:
Return status with generated report id and detection summary.
Notification/Dashboard:
Send notification to municipal endpoint or show report in admin dashboard.
Reports aggregated into heatmaps and prioritized worklists.
Sample API examples
Endpoint: POST /api/reports/upload Request (multipart form):

image: binary file
metadata: JSON string: { "user_id": "uuid-or-null", "device_lat": 19.0760, "device_lon": 72.8777, "timestamp": "2026-07-15T10:22:00Z", "device_heading": 84.2, "device_height_m": 1.5, "camera_fov_deg": 68 }
Response: { "report_id":"rep_12345", "status":"processing", "detections_summary": null }

Later (after processing) GET /api/reports/rep_12345 returns: { "report_id":"rep_12345", "status":"completed", "detections":[ { "detection_id":"det_987", "class":"pothole", "confidence":0.94, "bbox":[120,320,80,60], "mask_url":"https://storage/.../det_987_mask.png", "severity_score":0.78, "severity_label":"severe", "area_m2":0.42 } ], "location": {"lat":19.0762, "lon":72.8779}, "image_url":"https://storage/.../rep_12345_orig.jpg", "created_at":"2026-07-15T10:22:02Z" }

Database Configuration and Schema
Recommendations:

Use PostgreSQL with PostGIS extension for geospatial indexing and queries.
Use object storage for binary blobs (images/videos); store only URLs in DB.
Use UUID primary keys for records.
Use indexing on created_at, status, and geospatial index for location.
Suggested tables (SQL CREATE examples):

-- 1) users (optional) CREATE TABLE users ( user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT, email TEXT, role TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT now() );

-- 2) reports CREATE TABLE reports ( report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID REFERENCES users(user_id), image_url TEXT NOT NULL, thumbnail_url TEXT, reported_at TIMESTAMP WITH TIME ZONE NOT NULL, processed_at TIMESTAMP WITH TIME ZONE, status TEXT CHECK (status IN ('processing','completed','rejected')) DEFAULT 'processing', geom GEOGRAPHY(POINT, 4326), -- PostGIS geography point device_meta JSONB, notes TEXT );

-- index for spatial queries: CREATE INDEX idx_reports_geom ON reports USING GIST (geom);

-- 3) detections CREATE TABLE detections ( detection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), report_id UUID REFERENCES reports(report_id) ON DELETE CASCADE, class TEXT, confidence FLOAT, bbox JSONB, -- e.g., {"x":120,"y":320,"w":80,"h":60} mask_url TEXT, -- optional area_m2 FLOAT, severity_score FLOAT, severity_label TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT now() ); CREATE INDEX idx_detections_report ON detections(report_id);

-- 4) audit/status history (optional) CREATE TABLE report_history ( id SERIAL PRIMARY KEY, report_id UUID REFERENCES reports(report_id), event_ts TIMESTAMPTZ DEFAULT now(), event_type TEXT, details JSONB );

Storage recommendations:

Object Storage path layout: /reports/{report_id}/original.jpg /reports/{report_id}/thumbnail.jpg /reports/{report_id}/detections/{detection_id}_mask.png
Use signed URLs or pre-signed PUT for uploads.
Severity Estimation Method
Two practical approaches:

A. Heuristic (fast, no depth):

severity_score = area_pixels / reference_pixels
Convert area in pixels to meters² using calibration: area_m2 = area_pixels * 
(
s
e
n
s
o
r
w
i
d
t
h
/
i
m
a
g
e
w
i
d
t
h
)
∗
d
i
s
t
a
n
c
e
/
f
o
c
a
l
l
e
n
g
t
h
^2
If device height is known (phone held at 1.5 m), rough calibration is possible.
B. Learned regression (recommended for accuracy):

Train a small CNN/regressor that takes detected crop (and optionally monocular depth features) to predict depth/volume or severity grade, using labeled data (volume labels or severity categories).
Severity labels example:

minor (score < 0.3)
moderate (0.3 <= score < 0.6)
severe (score >= 0.6)
Handling Duplicates, Data Quality & Privacy
Deduplication: cluster reports by location/time (within radius and time window) using spatial queries and merge similar detections.
Quality: enforce minimum GPS accuracy threshold; allow manual verification.
Privacy: Blur faces/number plates before storing or require user consent. Store minimal personal data and comply with local laws.
End Results / Deliverables
Primary deliverables the system produces:

Single-report JSON: full detection payload for each upload.
Aggregated report CSV: daily/weekly list of confirmed potholes with severity and coordinates.
Dashboard:
Map with markers (severity-coded).
Heatmap layer of pothole density.
Worklist: prioritized tasks sorted by severity & traffic impact.
Images & evidence per report.
API endpoints for municipal systems to retrieve prioritized reparations list.
Alerts/Notifications: threshold-based triggers (e.g., >N severe potholes in area).
Analytics: counts, repair turnaround time, hotspot detection, model performance logs.
Evaluation Metrics & Monitoring
Model performance:

Detection: mean Average Precision (mAP@0.5 and mAP@[.5:.95]), precision, recall.
Segmentation: mean IoU (Intersection over Union).
Severity/regression: RMSE for volume prediction, classification accuracy for severity labels. Operational:
End-to-end latency (ms).
Throughput (requests/sec).
False positive rate — important to minimize manual workload.
Dashboard KPIs: reports processed per day, repairs scheduled, median time to repair.
Monitoring:

Logging for inference requests, latencies, failures.
Periodic retraining pipeline with new labeled data (active learning).
Drift detection for model input distribution over time.
Deployment & Scalability
Containerize services (Docker); orchestrate with Kubernetes for scale.
Autoscale inference pods based on queue length / CPU/GPU usage.
Use GPU instances for heavy inference/ batch processing; TI or CPU TF-Lite for edge devices.
CDN and caching for frequently requested images.
Backup DB and object storage; use point-in-time recovery for Postgres.
Security & Access Control
API authentication (JWT / OAuth).
Rate limiting to avoid abuse.
Signed image upload URLs; access-control on object storage.
Data encryption at rest (S3) and in transit (HTTPS).
Advantages, Benefits & Limitations
Advantages:

Fast, scalable reporting and prioritization.
Crowd-sourced coverage increases detection frequency.
Spatial analytics to allocate municipal resources.
Limitations:

Detection accuracy depends on quality of images and GPS accuracy.
Severity estimation from monocular images is approximate without calibration.
Privacy and false positives require human validation for high-stakes actions.
Future Work & Extensions
Integrate video-frame temporal consistency to improve detection.
Use stereo or LiDAR-equipped vehicles for accurate depth/volume measurement.
Integrate with municipal work-order systems for repair tracking.
Implement federated learning for privacy-aware model improvement.
References (selection)
Reference project: Gesture-Classification-System — https://github.com/Yashsmakwana/Gesture-Classification-System
Redmon, J. et al. (YOLO) — (original YOLO paper) https://pjreddie.com/darknet/yolo/
Bochkovskiy, A., Wang, C.-Y., Liao, H.-Y.M. (YOLOv4) https://arxiv.org/abs/2004.10934
Ultralytics YOLO (practical implementations): https://docs.ultralytics.com/
UNet paper (Segmentation) — Olaf Ronneberger et al.
PostGIS documentation — https://postgis.net/docs/
MobileNet and EfficientNet references for mobile backbones.
(When you prepare final submission, ensure proper citation formatting and add any other academic references used in training datasets or severity estimation methods.)

Appendix A — Example end-to-end JSON report (full)
{ "report_id":"rep_12345", "user_id":"user_789", "reported_at":"2026-07-15T10:22:00Z", "location":{"lat":19.0762,"lon":72.8779,"gps_accuracy_m":5.0}, "image_url":"https://storage/.../rep_12345/original.jpg", "detections":[ { "detection_id":"det_987", "class":"pothole", "confidence":0.94, "bbox":[120,320,80,60], "mask_url":"https://storage/.../rep_12345/det_987_mask.png", "area_m2":0.42, "severity_score":0.78, "severity_label":"severe" } ], "status":"completed", "notes":"user-provided note string here" }

Appendix B — Suggested Milestones to implement from this doc
Data collection plan & labeling schema.
Prototype detection model using YOLOv5 or YOLOv8 (transfer learning).
Backend API + object storage + minimal DB (reports + detections).
Simple dashboard showing incoming reports on a map.
Severity estimation heuristic; then upgrade to learned regressor.
Deploy scalability and monitoring.
