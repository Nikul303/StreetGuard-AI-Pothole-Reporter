# SYSTEM ARCHITECTURE & EXECUTION FLOW

## 5.1 Complete System Architecture

### 5.1.1 High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    StreetGuard AI Pothole Detection System              │
│                           End-to-End Pipeline                           │
└─────────────────────────────────────────────────────────────────────────┘

1. DATA ACQUISITION LAYER
   ┌──────────────────────────────────┐
   │  Input Sources:                  │
   │  ├─ Mobile App (User captures)   │
   │  ├─ Dashcam Feeds (24/7)         │
   │  ├─ Government Surveys           │
   │  ├─ Satellite Imagery            │
   │  └─ Drone Footage                │
   └────────────┬─────────────────────┘
                │
                ▼
2. PREPROCESSING & STORAGE
   ┌──────────────────────────────────┐
   │  Image Validation & Storage:     │
   │  ├─ Format Check (JPEG/PNG)      │
   │  ├─ Size Normalization (224x224) │
   │  ├─ Quality Assessment           │
   │  ├─ Duplicate Detection          │
   │  └─ Cloud Storage (AWS S3)       │
   └────────────┬─────────────────────┘
                │
                ▼
3. AI MODEL INFERENCE LAYER
   ┌──────────────────────────────────┐
   │  Deep Learning Pipeline:         │
   │  ├─ Model: ResNet50/EfficientNet │
   │  ├─ Input: Preprocessed Image    │
   │  ├─ Feature Extraction           │
   │  ├─ Classification Head          │
   │  └─ Output: Confidence Score     │
   └────────────┬─────────────────────┘
                │
                ▼
4. POST-PROCESSING & ANALYSIS
   ┌──────────────────────────────────┐
   │  Results Refinement:             │
   │  ├─ Threshold Application        │
   │  ├─ Bounding Box Generation      │
   │  ├─ Severity Classification      │
   │  ├─ Location Mapping (GPS)       │
   │  └─ Confidence Scoring           │
   └────────────┬─────────────────────┘
                │
                ▼
5. DATABASE & TRACKING
   ┌──────────────────────────────────┐
   │  Data Storage & Management:      │
   │  ├─ Pothole Records (PostgreSQL) │
   │  ├─ Location Index (GIS)         │
   │  ├─ Historical Data              │
   │  ├─ Maintenance Status           │
   │  └─ Analytics Cache (Redis)      │
   └────────────┬─────────────────────┘
                │
                ▼
6. VISUALIZATION & API LAYER
   ┌──────────────────────────────────┐
   │  Output & Integration:           │
   │  ├─ Web Dashboard (React)        │
   │  ├─ Mobile App                   │
   │  ├─ REST API (FastAPI)           │
   │  ├─ Real-time Notifications      │
   │  └─ Report Generation            │
   └────────────┬─────────────────────┘
                │
                ▼
7. DEPLOYMENT & MONITORING
   ┌──────────────────────────────────┐
   │  Production Infrastructure:      │
   │  ├─ Docker Containers            │
   │  ├─ Kubernetes Orchestration     │
   │  ├─ CI/CD Pipeline               │
   │  ├─ Health Monitoring            │
   │  └─ Auto-scaling                 │
   └──────────────────────────────────┘
```

---

## 5.2 Component-Level Architecture

### 5.2.1 Backend Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND SERVICES                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   API Gateway        │  (Request Routing & Load Balancing)
│   (FastAPI/Flask)    │  Port: 8000
└──────────┬───────────┘
           │
    ┌──────┼──────────────────────────────┐
    │      │                              │
    ▼      ▼                              ▼
┌────────────────┐  ┌─────────────────┐  ┌──────────────┐
│  Image Service │  │ Detection       │  │ Report       │
│  - Upload      │  │ Service         │  │ Generation   │
│  - Validate    │  │ - Inference     │  │ Service      │
│  - Store       │  │ - Classify      │  │ - Aggregate  │
│  - Retrieve    │  │ - Score         │  │ - Export     │
└────────┬───────┘  └────────┬────────┘  └──────┬───────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────┐
    │         DATABASE LAYER                          │
    ├─────────────────────────────────────────────────┤
    │                                                 │
    │  ┌──────────────────────────────────────────┐  │
    │  │ PostgreSQL (Primary Data Store)          │  │
    │  │ • Pothole Records                        │  │
    │  │ • User Reports                           │  │
    │  │ • Repair History                         │  │
    │  │ • Maintenance Schedule                   │  │
    │  └──────────────────────────────────────────┘  │
    │                                                 │
    │  ┌──────────────────────────────────────────┐  │
    │  │ Redis (Cache Layer)                      │  │
    │  │ • Session Management                     │  │
    │  │ • Query Results Cache                    │  │
    │  │ • Real-time Notifications Queue          │  │
    │  └──────────────────────────────────────────┘  │
    │                                                 │
    │  ┌──────────────────────────────────────────┐  │
    │  │ Elasticsearch (Search & Analytics)       │  │
    │  │ • Full-text Search on Reports            │  │
    │  │ • Analytics Indexing                     │  │
    │  │ • Time-series Data                       │  │
    │  └──────────────────────────────────────────┘  │
    │                                                 │
    │  ┌──────────────────────────────────────────┐  │
    │  │ S3/Cloud Storage (Blob Store)            │  │
    │  │ • Original Images                        │  │
    │  │ • Processed Images                       │  │
    │  │ • Model Files                            │  │
    │  │ • Backups                                │  │
    │  └──────────────────────────────────────────┘  │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

### 5.2.2 ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL INFERENCE PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

INPUT IMAGE
    │
    ▼
┌─────────────────────────────────────┐
│  Preprocessing Module               │
├─────────────────────────────────────┤
│  1. Image Loading                   │
│     ├─ Read from disk/cloud         │
│     ├─ Verify format & integrity    │
│     └─ Load into memory             │
│                                     │
│  2. Normalization                   │
│     ├─ Resize to 224×224            │
│     ├─ Convert RGB color space      │
│     └─ Normalize pixel values       │
│        (subtract mean, divide std)  │
│                                     │
│  3. Augmentation (Optional)         │
│     ├─ Horizontal flip              │
│     ├─ Random rotation              │
│     ├─ Brightness adjustment        │
│     └─ Contrast enhancement         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Normalized Image (224×224×3)       │
│  Shape: [1, 224, 224, 3]            │
│  Values: 0.0 to 1.0                 │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  CNN Model (ResNet50)               │
├─────────────────────────────────────┤
│  Stage 1: Feature Extraction        │
│  ├─ Conv layers (7×7 → 3×3 filters) │
│  ├─ Edge & texture detection        │
│  ├─ Output: [1, 112, 112, 64]       │
│  │                                  │
│  Stage 2: Pooling & Refinement      │
│  ├─ Max pooling (reduce spatial)    │
│  ├─ Output: [1, 56, 56, 64]         │
│  │                                  │
│  Stage 3: Residual Blocks           │
│  ├─ ResBlock 1-4 (Progressive)      │
│  ├─ Feature dimension: 64→256→512→  │
│  ├─ Output: [1, 7, 7, 2048]         │
│  │                                  │
│  Stage 4: Global Pooling            │
│  ├─ Average pooling over space      │
│  └─ Output: [1, 2048]               │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Classification Head                │
├─────────────────────────────────────┤
│  Dense Layer 1:                     │
│  ├─ Input: 2048                     │
│  ├─ Neurons: 512                    │
│  ├─ Activation: ReLU                │
│  ├─ Dropout: 0.5                    │
│  └─ Output: [1, 512]                │
│                                     │
│  Dense Layer 2:                     │
│  ├─ Input: 512                      │
│  ├─ Neurons: 256                    │
│  ├─ Activation: ReLU                │
│  ├─ Dropout: 0.5                    │
│  └─ Output: [1, 256]                │
│                                     │
│  Output Layer (Binary):             │
│  ├─ Input: 256                      │
│  ├─ Neurons: 2                      │
│  ├─ Activation: Softmax             │
│  └─ Output: [P(no_pothole), P(pot)] │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Post-Processing                    │
├─────────────────────────────────────┤
│  1. Probability Extraction          │
│     ├─ Pothole score: P(pothole)    │
│     └─ Confidence: max(P)           │
│                                     │
│  2. Threshold Application           │
│     ├─ If confidence > 0.7:         │
│     │  └─ POSITIVE (Pothole)        │
│     └─ Else:                        │
│        └─ NEGATIVE (No Pothole)     │
│                                     │
│  3. Metadata Generation             │
│     ├─ Timestamp                    │
│     ├─ Location (GPS)               │
│     ├─ Severity Score               │
│     ├─ Bounding Box                 │
│     └─ Confidence Level             │
└────────────┬────────────────────────┘
             │
             ▼
OUTPUT RESULT
├─ Detection: True/False
├─ Confidence: 0.0-1.0
├─ Severity: Low/Medium/High
├─ Location: Latitude, Longitude
├─ Timestamp: ISO 8601
└─ Metadata: Image ID, User, Source
```

---

## 5.3 Input/Output Execution Flow

### 5.3.1 Complete Request-Response Cycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLIENT-SERVER INTERACTION                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           USER/CLIENT SIDE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. USER ACTION                                                         │
│  ├─ Mobile app: Capture image using camera                             │
│  ├─ Web app: Upload image via drag-drop                                │
│  ├─ Dashcam: Automatic streaming                                       │
│  └─ Government: Batch upload                                           │
│                                                                         │
│  2. LOCAL PROCESSING (Optional)                                         │
│  ├─ Compress image (reduce bandwidth)                                  │
│  ├─ Get GPS coordinates (mobile)                                       │
│  ├─ Capture metadata (timestamp, device)                               │
│  └─ Create multipart request                                           │
│                                                                         │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
         HTTP POST /api/detect
         ─────────────────────────────────────────
         Headers:
         ├─ Content-Type: multipart/form-data
         ├─ Authorization: Bearer {token}
         └─ User-Agent: {app_version}
         
         Body:
         ├─ image: (binary file data)
         ├─ latitude: {float}
         ├─ longitude: {float}
         ├─ source: "mobile_app"
         └─ user_id: {uuid}
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND SERVER SIDE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: REQUEST HANDLING (API Gateway)                                 │
│  ├─ Receive HTTP request                                               │
│  ├─ Validate authentication token                                      │
│  ├─ Check rate limits                                                  │
│  ├─ Log request metadata                                               │
│  └─ Route to detection service                                         │
│                                                                         │
│  STEP 2: IMAGE VALIDATION                                               │
│  ├─ Check file type (JPEG/PNG)                                         │
│  ├─ Verify file size (1-50 MB)                                         │
│  ├─ Scan for malware                                                   │
│  ├─ Validate dimensions (min 224×224)                                  │
│  └─ Generate unique image ID: uuid4()                                  │
│                                                                         │
│  STEP 3: STORAGE & INDEXING                                             │
│  ├─ Save to S3: s3://streetguard/images/{image_id}/original.jpg        │
│  ├─ Store metadata in Redis:                                           │
│  │  Key: image:{image_id}                                              │
│  │  Fields:                                                            │
│  │  ├─ status: "processing"                                            │
│  │  ├─ timestamp: {iso_timestamp}                                      │
│  │  ├─ user_id: {user_id}                                              │
│  │  ├─ coordinates: {lat},{lon}                                        │
│  │  └─ ttl: 86400s (24 hours)                                          │
│  │                                                                     │
│  │  Create database record:                                            │
│  │  INSERT INTO images (id, user_id, latitude, longitude, source)     │
│  │  VALUES ({image_id}, {user_id}, {lat}, {lon}, 'mobile_app')        │
│  │                                                                     │
│  └─ Enqueue for processing                                             │
│                                                                         │
│  STEP 4: ML INFERENCE PREPARATION                                       │
│  ├─ Load image from storage                                            │
│  ├─ Resize to 224×224 pixels                                           │
│  ├─ Normalize pixel values (0-1 range)                                 │
│  ├─ Create batch tensor: shape [1, 224, 224, 3]                       │
│  └─ Load pre-trained model to GPU (if available)                       │
│                                                                         │
│  STEP 5: MODEL INFERENCE (Critical Path)                                │
│  ├─ Forward pass through ResNet50:                                     │
│  │                                                                     │
│  │  Input: [1, 224, 224, 3] tensor                                     │
│  │    │                                                                │
│  │    └─ Timing: ~50-100ms on GPU                                      │
│  │                                                                     │
│  │  Process:                                                           │
│  │  ├─ Feature extraction (Conv layers)                                │
│  │  ├─ Residual blocks (ResBlock 1-4)                                  │
│  │  ├─ Global pooling                                                  │
│  │  ├─ Dense layers with dropout                                       │
│  │  └─ Softmax normalization                                           │
│  │                                                                     │
│  │  Output: [0.15, 0.85]                                               │
│  │  ├─ P(No Pothole) = 0.15                                            │
│  │  └─ P(Pothole) = 0.85                                               │
│  │                                                                     │
│  ├─ Extract confidence: max(0.15, 0.85) = 0.85                         │
│  ├─ Calculate uncertainty: std(logits)                                 │
│  └─ Save timing: model_inference_ms = 67                               │
│                                                                         │
│  STEP 6: POST-PROCESSING                                                │
│  ├─ Apply threshold (0.70):                                            │
│  │  ├─ Is 0.85 > 0.70? YES                                             │
│  │  └─ Prediction: POTHOLE DETECTED                                    │
│  │                                                                     │
│  ├─ Calculate severity:                                                │
│  │  ├─ If confidence > 0.9: "HIGH"                                     │
│  │  ├─ If 0.7-0.9: "MEDIUM"                                            │
│  │  └─ If 0.5-0.7: "LOW"                                               │
│  │  └─ Result: "MEDIUM" (0.85)                                         │
│  │                                                                     │
│  ├─ Generate bounding box (if applicable):                             │
│  │  ├─ X_min: 40, Y_min: 50                                            │
│  │  ├─ X_max: 185, Y_max: 200                                          │
│  │  └─ Width: 145px, Height: 150px                                     │
│  │                                                                     │
│  └─ Create result object                                               │
│                                                                         │
│  STEP 7: DATABASE STORAGE                                               │
│  ├─ Insert detection record:                                           │
│  │  INSERT INTO detections (                                           │
│  │    image_id, is_pothole, confidence, severity,                      │
│  │    bounding_box, latitude, longitude, timestamp, model_version      │
│  │  ) VALUES (                                                         │
│  │    '{uuid}', true, 0.85, 'MEDIUM',                                  │
│  │    '{40,50,185,200}', 28.6139, 77.2090,                             │
│  │    NOW(), 'resnet50_v2.0'                                           │
│  │  )                                                                  │
│  │                                                                     │
│  ├─ Update cache:                                                      │
│  │  GET {image_id}:detection_result →                                  │
│  │  {                                                                  │
│  │    "is_pothole": true,                                              │
│  │    "confidence": 0.85,                                              │
│  │    "severity": "MEDIUM",                                            │
│  │    "processed_at": "2024-07-15T10:30:45Z"                           │
│  │  }                                                                  │
│  │                                                                     │
│  └─ Log metrics:                                                       │
│     ├─ Inference time: 67ms                                            │
│     ├─ Model version: resnet50_v2.0                                    │
│     ├─ GPU utilization: 34%                                            │
│     └─ Queue wait time: 2.3s                                           │
│                                                                         │
│  STEP 8: ASYNC OPERATIONS (Background)                                  │
│  ├─ Generate notification:                                             │
│  │  └─ Publish to Kafka: topic="pothole_detected"                      │
│  │     Message: {image_id, severity, location}                        │
│  │                                                                     │
│  ├─ Index for search:                                                  │
│  │  └─ Push to Elasticsearch with severity, location                   │
│  │                                                                     │
│  ├─ Update geospatial index:                                           │
│  │  └─ Store in PostGIS for location queries                           │
│  │                                                                     │
│  ├─ Process image for archive:                                         │
│  │  ├─ Create thumbnail (100×100)                                      │
│  │  ├─ Generate heatmap overlay                                        │
│  │  └─ Save to S3                                                      │
│  │                                                                     │
│  └─ Trigger notifications:                                             │
│     ├─ SMS to municipality (if HIGH severity)                          │
│     ├─ Push notification to nearby users                               │
│     └─ Email digest (daily)                                            │
│                                                                         │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
         HTTP 200 OK (Response)
         ───────────────────────────────────────────
         Content-Type: application/json
         
         {
           "success": true,
           "image_id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
           "detection": {
             "is_pothole": true,
             "confidence": 0.85,
             "confidence_percentage": "85%",
             "severity": "MEDIUM",
             "bounding_box": {
               "x_min": 40,
               "y_min": 50,
               "x_max": 185,
               "y_max": 200,
               "width": 145,
               "height": 150
             }
           },
           "location": {
             "latitude": 28.6139,
             "longitude": 77.2090,
             "address": "New Delhi, India",
             "nearby_reports": 3
           },
           "metadata": {
             "model": "resnet50_v2.0",
             "model_accuracy": "95%",
             "inference_time_ms": 67,
             "processed_at": "2024-07-15T10:30:45.123Z",
             "queue_wait_ms": 2300
           },
           "actions": {
             "view_result": "/results/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
             "report_false_positive": "/api/feedback/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6/negative",
             "share": "/share/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
           }
         }
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER/CLIENT SIDE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  3. RESPONSE HANDLING                                                    │
│  ├─ Receive JSON response                                              │
│  ├─ Parse result (pothole detected: YES)                               │
│  ├─ Display alert: "⚠️ Pothole Found! (85% confidence)"                │
│  ├─ Show location on map                                               │
│  ├─ Enable actions (Report, Share, View Details)                       │
│  └─ Log user interaction                                               │
│                                                                         │
│  4. USER ACTION (Post-Detection)                                         │
│  ├─ Option A: Confirm Detection (Feedback)                             │
│  │  └─ POST /api/feedback {image_id, feedback: "positive"}             │
│  │                                                                     │
│  ├─ Option B: Mark False Positive                                      │
│  │  └─ POST /api/feedback {image_id, feedback: "negative"}             │
│  │  └─ This data retrains the model                                    │
│  │                                                                     │
│  ├─ Option C: Report to Municipality                                   │
│  │  └─ POST /api/report {image_id, priority: "high"}                   │
│  │                                                                     │
│  └─ Option D: Share with Others                                        │
│     └─ Generate shareable link: /share/{image_id}                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3.2 Batch Processing Flow (Government Surveys)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING PIPELINE                            │
│          (For processing multiple images/videos in parallel)            │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Batch of 10,000 road images
       │
       ▼
1. BATCH VALIDATION
   ├─ Total images: 10,000
   ├─ Valid format: 9,850 ✓
   ├─ Invalid format: 150 ✗
   ├─ Total size: 2.5 GB
   └─ Estimated time: ~150 minutes (at 66 images/min)

2. QUEUE DISTRIBUTION
   ├─ Create task groups (8 groups × 1,250 images)
   ├─ Distribute to 8 GPU workers
   ├─ Each worker: 1 GPU, 16 GB RAM
   ├─ Parallel processing
   └─ Expected completion: 30 minutes

3. WORKER PROCESSING (Per GPU)
   ├─ Batch size: 64 images at once
   ├─ Process 1,250 images
   ├─ ~20 inference batches
   ├─ Time per batch: 1.5 seconds
   ├─ Processing time: ~30 seconds
   └─ Write results in parallel

4. RESULT AGGREGATION
   ├─ Collect results from 8 workers
   ├─ Total potholes detected: ~1,200 (12%)
   ├─ High severity: 180 (1.8%)
   ├─ Medium severity: 600 (6%)
   ├─ Low severity: 420 (4.2%)
   ├─ Generate summary report
   └─ Create heatmap

5. OUTPUT GENERATION
   ├─ CSV Export: results.csv (all detections)
   ├─ GeoJSON: pothole_locations.geojson (for mapping)
   ├─ Heatmap: severity_heatmap.png (visualize hotspots)
   ├─ Report: batch_report_20240715.pdf
   └─ Database entries: 10,000 records inserted

6. NOTIFICATION
   ├─ Email: "Batch processing complete. 1,200 potholes detected."
   ├─ Dashboard: Update statistics
   ├─ API: Make results available via /api/batch/{batch_id}
   └─ Archive: Store in long-term storage
```

---

## 5.4 Database Configuration & Schema

### 5.4.1 PostgreSQL Schema (Primary Database)

```sql
-- Core Tables

-- 1. USERS TABLE
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(500) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(15),
    user_type ENUM('citizen', 'inspector', 'admin', 'municipality') NOT NULL,
    organization VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    CONSTRAINT check_email_format CHECK (email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

-- 2. IMAGES TABLE
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,  -- S3 path: s3://bucket/images/{id}/original.jpg
    file_size_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    location_address VARCHAR(500),
    source ENUM('mobile_app', 'dashcam', 'government_survey', 'manual_upload') NOT NULL,
    device_info VARCHAR(500),  -- e.g., "iPhone 13, iOS 15.2"
    is_processed BOOLEAN DEFAULT false,
    processing_status VARCHAR(50),  -- 'queued', 'processing', 'completed', 'failed'
    processing_error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_location (latitude, longitude),
    INDEX idx_created_at (created_at)
);

-- 3. DETECTIONS TABLE (Main Output Table)
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    is_pothole BOOLEAN NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') NOT NULL,
    model_version VARCHAR(50) NOT NULL,  -- e.g., 'resnet50_v2.0'
    inference_time_ms INTEGER,
    bounding_box_x_min INTEGER,
    bounding_box_y_min INTEGER,
    bounding_box_x_max INTEGER,
    bounding_box_y_max INTEGER,
    bounding_box_width INTEGER,
    bounding_box_height INTEGER,
    gps_latitude DECIMAL(10, 8),
    gps_longitude DECIMAL(11, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at_epoch BIGINT GENERATED ALWAYS AS (EXTRACT(EPOCH FROM created_at)) STORED,
    INDEX idx_image_id (image_id),
    INDEX idx_confidence (confidence),
    INDEX idx_severity (severity),
    INDEX idx_is_pothole (is_pothole)
);

-- 4. FEEDBACK TABLE (For Model Retraining)
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feedback_type ENUM('positive', 'negative', 'unclear') NOT NULL,
    comment TEXT,
    is_used_for_training BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_image_id (image_id),
    INDEX idx_user_id (user_id),
    INDEX idx_is_used (is_used_for_training)
);

-- 5. MAINTENANCE_RECORDS TABLE
CREATE TABLE maintenance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    status ENUM('reported', 'assigned', 'in_progress', 'completed', 'closed') NOT NULL,
    priority ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    assigned_to_inspector_id UUID REFERENCES users(id) ON DELETE SET NULL,
    estimated_cost DECIMAL(10, 2),
    actual_cost DECIMAL(10, 2),
    repair_notes TEXT,
    before_image_path VARCHAR(1000),
    after_image_path VARCHAR(1000),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_assigned_to (assigned_to_inspector_id)
);

-- 6. MODEL_METRICS TABLE (For tracking model performance)
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL,
    metric_date DATE NOT NULL,
    total_inferences INTEGER NOT NULL DEFAULT 0,
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    avg_inference_time_ms DECIMAL(10, 2),
    avg_gpu_utilization DECIMAL(5, 2),
    avg_memory_usage_mb INTEGER,
    inference_errors INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_version, metric_date),
    INDEX idx_model_version (model_version),
    INDEX idx_metric_date (metric_date)
);

-- 7. API_LOGS TABLE (For auditing and performance tracking)
CREATE TABLE api_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    ip_address INET,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_endpoint (endpoint),
    INDEX idx_status_code (status_code),
    INDEX idx_created_at (created_at)
);
```

### 5.4.2 Redis Cache Configuration

```
Redis Configuration:
─────────────────────

Server Settings:
├─ Host: redis.streetguard.prod
├─ Port: 6379
├─ Database: 0
├─ Memory Limit: 16 GB
├─ Eviction Policy: allkeys-lru (Least Recently Used)
└─ Persistence: RDB snapshots every 1 hour

Key-Value Patterns:
─────────────────────

1. Image Cache
   Key: image:{image_id}
   Value: {
     "status": "completed",
     "user_id": "uuid",
     "latitude": "28.6139",
     "longitude": "77.2090",
     "timestamp": "2024-07-15T10:30:45Z",
     "processed_at": "2024-07-15T10:31:52Z"
   }
   TTL: 86400 seconds (24 hours)

2. Detection Results Cache
   Key: detection:{detection_id}
   Value: {
     "is_pothole": true,
     "confidence": 0.85,
     "severity": "MEDIUM",
     "bounding_box": {...},
     "model_version": "resnet50_v2.0"
   }
   TTL: 604800 seconds (7 days)

3. Session Management
   Key: session:{session_id}
   Value: {
     "user_id": "uuid",
     "email": "user@example.com",
     "user_type": "inspector",
     "last_activity": "2024-07-15T10:30:45Z"
   }
   TTL: 3600 seconds (1 hour)

4. Rate Limiting
   Key: ratelimit:{user_id}:{endpoint}
   Value: request_count (integer)
   TTL: 60 seconds
   Limit: 1000 requests per minute per user

5. Processing Queue
   Key: queue:pending_images
   Type: List (FIFO)
   Values: [image_id_1, image_id_2, ...]
   
6. Statistics Cache
   Key: stats:{statistic_type}
   Value: {
     "total_detections": 1523,
     "potholes_detected": 1200,
     "high_severity": 180,
     "last_updated": "2024-07-15T10:30:00Z"
   }
   TTL: 3600 seconds (1 hour)

7. User Analytics
   Key: analytics:user:{user_id}:daily
   Value: {
     "uploads_count": 42,
     "detections_count": 35,
     "date": "2024-07-15"
   }
   TTL: 2592000 seconds (30 days)
```

### 5.4.3 Geospatial Database (PostGIS Extension)

```
PostGIS Configuration:
──────────────────────

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Geospatial Detections Table
CREATE TABLE detections_geo AS
SELECT 
    id,
    image_id,
    severity,
    confidence,
    ST_Point(gps_longitude, gps_latitude) AS location_point,
    ST_GeomFromText('POINT(' || gps_longitude || ' ' || gps_latitude || ')', 4326) AS location_geometry,
    created_at
FROM detections;

-- Create spatial index for faster queries
CREATE INDEX idx_detections_location 
ON detections_geo 
USING GIST(location_geometry);

-- Useful Queries:

-- 1. Find all potholes within 5 km radius
SELECT *
FROM detections_geo
WHERE ST_DWithin(
    location_geometry,
    ST_Point(77.2090, 28.6139)::geography,
    5000  -- 5 km in meters
)
AND severity IN ('HIGH', 'CRITICAL')
ORDER BY created_at DESC;

-- 2. Cluster potholes into heatmap zones
SELECT
    ST_ClusterKMeans(location_geometry, 50) OVER () AS cluster_id,
    severity,
    COUNT(*) AS count,
    ST_Centroid(ST_Collect(location_geometry)) AS center
FROM detections_geo
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY cluster_id, severity;

-- 3. Find intersection of potholes and road networks
SELECT 
    d.*,
    r.road_name,
    r.road_category
FROM detections_geo d
JOIN roads r ON ST_Intersects(d.location_geometry, r.geom)
WHERE r.city = 'New Delhi'
ORDER BY d.confidence DESC;
```

### 5.4.4 Elasticsearch Configuration (Search & Analytics)

```
Elasticsearch Mapping:
──────────────────────

PUT /pothole_detections
{
  "settings": {
    "number_of_shards": 5,
    "number_of_replicas": 2,
    "index.refresh_interval": "30s",
    "analysis": {
      "analyzer": {
        "location_analyzer": {
          "type": "standard",
          "stopwords": "_english_"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "detection_id": {
        "type": "keyword"
      },
      "image_id": {
        "type": "keyword"
      },
      "is_pothole": {
        "type": "boolean"
      },
      "confidence": {
        "type": "float"
      },
      "severity": {
        "type": "keyword"
      },
      "location": {
        "type": "geo_point"
      },
      "address": {
        "type": "text",
        "analyzer": "location_analyzer"
      },
      "timestamp": {
        "type": "date"
      },
      "model_version": {
        "type": "keyword"
      },
      "user_id": {
        "type": "keyword"
      },
      "feedback_count": {
        "type": "integer"
      },
      "maintenance_status": {
        "type": "keyword"
      }
    }
  }
}

-- Sample Document
{
  "detection_id": "uuid-1",
  "image_id": "uuid-2",
  "is_pothole": true,
  "confidence": 0.85,
  "severity": "MEDIUM",
  "location": {
    "lat": 28.6139,
    "lon": 77.2090
  },
  "address": "New Delhi, India",
  "timestamp": "2024-07-15T10:30:45Z",
  "model_version": "resnet50_v2.0",
  "user_id": "uuid-3",
  "feedback_count": 5,
  "maintenance_status": "assigned"
}

-- Search Queries

-- 1. Full-text search by location
GET /pothole_detections/_search
{
  "query": {
    "match": {
      "address": "New Delhi"
    }
  },
  "sort": [
    {
      "confidence": {"order": "desc"}
    }
  ]
}

-- 2. Filter by severity and location radius
GET /pothole_detections/_search
{
  "query": {
    "bool": {
      "must": [
        {"term": {"severity": "HIGH"}},
        {"range": {"confidence": {"gte": 0.8}}}
      ],
      "filter": [
        {"geo_distance": {
          "distance": "5km",
          "location": {
            "lat": 28.6139,
            "lon": 77.2090
          }
        }}
      ]
    }
  }
}

-- 3. Time-series aggregation
GET /pothole_detections/_search
{
  "aggs": {
    "potholes_by_day": {
      "date_histogram": {
        "field": "timestamp",
        "interval": "day"
      },
      "aggs": {
        "avg_confidence": {
          "avg": {"field": "confidence"}
        },
        "severity_breakdown": {
          "terms": {"field": "severity"}
        }
      }
    }
  }
}
```

### 5.4.5 Database Connection Pooling

```
Connection Pool Configuration:
─────────────────────────────

1. PostgreSQL Connection Pool (pgBouncer)
   ├─ Pool Mode: transaction (most efficient)
   ├─ Max Connections: 1000
   ├─ Max DB Connections: 500
   ├─ Min Pool Size: 50
   ├─ Max Pool Size: 250
   ├─ Connection Timeout: 30s
   ├─ Idle Timeout: 600s
   ├─ Server Lifetime: 3600s
   └─ Query Timeout: 180s

2. Redis Connection Pool
   ├─ Min Connections: 10
   ├─ Max Connections: 100
   ├─ Connection Timeout: 5s
   ├─ Socket Timeout: 5s
   ├─ Idle Timeout: 300s
   └─ Retry Attempts: 3

3. Connection Monitoring
   ├─ Active Connections: monitored every 10s
   ├─ Alert if > 80% of max capacity
   ├─ Alert if avg query time > 100ms
   ├─ Auto-scaling based on CPU/Memory
   └─ Connection leak detection
```

---

## 5.5 End-to-End Execution Timeline

### 5.5.1 Single Image Processing Timeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROCESSING TIMELINE (Single Image)                   │
└─────────────────────────────────────────────────────────────────────────┘

Time    Event                           Duration    Cumulative
──────  ──────────────────────────────  ─────────   ───────────

T+0ms   User submits image              -           0ms
        ├─ Image file: 2.5 MB
        ├─ Format: JPEG
        └─ Resolution: 2048×2048

T+0-50ms   Request reaches server       50ms        50ms
        ├─ HTTP request parsing: 10ms
        ├─ Authentication check: 15ms
        ├─ Authorization check: 15ms
        └─ Rate limit check: 10ms

T+50-150ms   File validation & storage   100ms       150ms
        ├─ Virus scan: 30ms
        ├─ Format validation: 20ms
        ├─ S3 upload: 40ms
        └─ Database insert: 10ms

T+150-250ms   Image preprocessing        100ms       250ms
        ├─ Load from S3: 30ms
        ├─ Resize to 224×224: 20ms
        ├─ Normalize pixels: 30ms
        └─ Create tensor: 20ms

T+250-320ms   Model inference            70ms        320ms
        ├─ Forward pass (GPU): 55ms
        ├─ Output softmax: 10ms
        └─ Extract scores: 5ms

T+320-380ms   Post-processing            60ms        380ms
        ├─ Apply threshold: 10ms
        ├─ Calculate severity: 15ms
        ├─ Generate bounding box: 20ms
        └─ Prepare result: 15ms

T+380-420ms   Database operations        40ms        420ms
        ├─ Detection record insert: 20ms
        ├─ Cache update: 15ms
        └─ Index update: 5ms

T+420-480ms   Async operations           60ms        480ms
        ├─ Elasticsearch indexing: 25ms
        ├─ Notification queue: 15ms
        ├─ Analytics logging: 15ms
        └─ Geospatial indexing: 5ms

T+480-500ms   Response generation        20ms        500ms
        ├─ JSON serialization: 10ms
        ├─ HTTP response send: 10ms
        └─ Client receives: 0ms

                          TOTAL TIME: 500ms (0.5 seconds)

Performance Breakdown:
─────────────────────
- Network & I/O: 50 + 100 + 30 + 40 = 220ms (44%)
- ML Inference: 70ms (14%)
- Processing: 60 + 40 + 60 + 20 = 180ms (36%)
- Overhead: 30ms (6%)

Bottlenecks:
────────────
1. S3 operations (file upload/download): ~70ms
2. Database operations: ~60ms
3. Image preprocessing (resize, normalize): ~50ms

Optimizations Applied:
──────────────────────
✓ GPU acceleration for inference (70ms vs 500ms on CPU)
✓ Image preprocessing parallelization
✓ Batch database operations
✓ Redis caching for frequent queries
✓ Async operations for non-critical tasks
```

### 5.5.2 Batch Processing Timeline (10,000 Images)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  BATCH PROCESSING TIMELINE (10,000 Images)              │
└─────────────────────────────────────────────────────────────────────────┘

Phase       Task                        Duration    Workers  Total Time
─────────────────────────────────────────────────────────────────────────

SETUP       Batch validation            5 min       -        5 min
            ├─ Check 10,000 files
            ├─ Validate formats
            └─ Create task groups

            GPU worker initialization   3 min       8        3 min
            ├─ Load models (4 min each)
            ├─ Allocate memory
            └─ Set up connections

DATA        Upload to S3                15 min      1        15 min
            └─ 2.5GB @ 170 MB/s

PROCESSING  Distributed inference       25 min      8        3.5 min
            ├─ 10,000 ÷ 8 workers = 1,250/worker
            ├─ Batch size: 64 images
            ├─ ~20 batches per worker
            ├─ Time per batch: 1.5s
            └─ Total worker time: 30s each

RESULTS     Aggregate results           5 min       1        5 min
            ├─ Collect from 8 workers
            ├─ Database insert
            └─ Generate report

TOTAL TIME FOR 10,000 IMAGES: ~51 minutes

Metrics:
─────────
- Throughput: ~196 images/minute
- Per-image avg: 306ms (including I/O, batching overhead)
- GPU utilization: 85%
- Total cost: ~$2.50 (8 × GPU-hour @ $0.30/hour)
- Cost per image: $0.00025 (0.025 cents)
```

---

## 5.6 System Performance Metrics & Benchmarks

### 5.6.1 Expected Performance Characteristics

```
API Response Times:
───────────────────
- Single image detection: 500-800ms (P95)
- Batch query: 50-200ms
- Search query: 100-500ms
- Geospatial query: 200-1000ms

Model Performance:
───────────────────
- Accuracy: 95% (±2%)
- Precision: 93%
- Recall: 96%
- F1-Score: 0.945
- False positive rate: 7%
- False negative rate: 4%

Infrastructure Metrics:
──────────────────────
- API uptime: 99.95%
- Database uptime: 99.99%
- Model inference latency: 67ms (p50), 120ms (p95)
- Cache hit ratio: 75%
- Database query response: <100ms (p99)

Scaling Characteristics:
────────────────────────
- Handles 1000 concurrent users
- Processes 50 images/second at peak
- Supports 10M+ records in database
- Real-time detection (< 1 second per image)
```

