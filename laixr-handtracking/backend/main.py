import asyncio
import shutil
import sqlite3
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import json
from one_euro_filter_local import OneEuroFilter
from typing import Dict, Any, List, Tuple, Optional
from contextlib import asynccontextmanager
import os
import logging
import warnings
from mediapipe.framework.formats import landmark_pb2
import io
import csv
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime
from scipy.signal import butter, filtfilt
from pathlib import Path
from statistics import mean

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress known protobuf deprecation warning noise
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase.GetPrototype\(\) is deprecated"
)

DATABASE_URL = "laixr_handtracking_dev.db"
# Ensure paths are anchored to this file's directory
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
DATABASE_URL = os.path.join(BASE_DIR, "laixr_handtracking_dev.db")

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')

# --- Database Management ---
def db_execute(query, params=()):
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def db_query_one(query, params=()):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    return result

def db_query_all(query, params=()):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result

def init_db():
    # Migrate database if previously created in project root when started with different working dir
    try:
        project_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
        legacy_db = os.path.join(project_root, "laixr_handtracking_dev.db")
        if os.path.exists(legacy_db) and not os.path.exists(DATABASE_URL):
            shutil.copyfile(legacy_db, DATABASE_URL)
            logger.info(f"Migrated legacy database from {legacy_db} to {DATABASE_URL}")
    except Exception as e:
        logger.warning(f"Database migration check failed: {e}")
    db_execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        original_name TEXT,
        status TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        analysis_parameters TEXT,
        error_message TEXT,
        results TEXT,
        video_path TEXT,
        annotated_video_path TEXT
    )
    """)
    # Add columns if they don't exist to avoid errors on restart
    try:
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(analyses)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'raw_landmarks_json' not in columns:
            cursor.execute("ALTER TABLE analyses ADD COLUMN raw_landmarks_json TEXT")
            logger.info("Added 'raw_landmarks_json' column to 'analyses' table.")
        if 'video_path' not in columns:
            cursor.execute("ALTER TABLE analyses ADD COLUMN video_path TEXT")
            logger.info("Added 'video_path' column to 'analyses' table.")
        if 'original_landmarks_json' not in columns:
            cursor.execute("ALTER TABLE analyses ADD COLUMN original_landmarks_json TEXT")
            logger.info("Added 'original_landmarks_json' column to 'analyses' table.")
        if 'annotated_video_path' not in columns:
            cursor.execute("ALTER TABLE analyses ADD COLUMN annotated_video_path TEXT")
            logger.info("Added 'annotated_video_path' column to 'analyses' table.")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update database schema: {e}")

# --- Pydantic Models ---
class TimeRange(BaseModel):
    start_time: float
    end_time: float

class CorrectionPayload(BaseModel):
    time_ranges_to_remove: List[TimeRange]
    hand: str # 'Left' or 'Right'

# --- WebSocket Management for Live View ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, analysis_id: str):
        await websocket.accept()
        self.active_connections[analysis_id] = websocket

    def disconnect(self, analysis_id: str):
        if analysis_id in self.active_connections:
            del self.active_connections[analysis_id]

    async def send_frame(self, analysis_id: str, frame_data: bytes):
        if analysis_id in self.active_connections:
            websocket = self.active_connections[analysis_id]
            try:
                await websocket.send_bytes(frame_data)
            except (WebSocketDisconnect, ConnectionResetError):
                logger.info(f"Client for {analysis_id} disconnected during stream.")
                self.disconnect(analysis_id)

    async def close_connection(self, analysis_id: str):
        if analysis_id in self.active_connections:
            websocket = self.active_connections.pop(analysis_id)
            await websocket.close()
            logger.info(f"Closed WebSocket for analysis {analysis_id}")

manager = ConnectionManager()

# --- Analysis Core Logic ---
def detect_and_remove_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect and remove outliers from data using specified method.
    
    Args:
        data: Input data array
        method: 'iqr' for interquartile range, 'zscore' for z-score method
        threshold: Threshold for outlier detection
        
    Returns:
        tuple: (cleaned_data, outlier_mask)
    """
    if len(data) == 0:
        return data, np.array([], dtype=bool)
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outlier_mask = z_scores > threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Don't remove more than 10% of the data to preserve analysis integrity
    if np.sum(outlier_mask) > len(data) * 0.1:
        # If too many outliers, only remove the most extreme ones
        if method == 'iqr':
            # Use more conservative threshold
            conservative_threshold = threshold * 2
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - conservative_threshold * IQR
            upper_bound = Q3 + conservative_threshold * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        else:  # zscore
            conservative_threshold = threshold * 1.5
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            outlier_mask = z_scores > conservative_threshold
    
    return data[~outlier_mask], outlier_mask

def calculate_kinematics_and_metrics(
    hand_df: pd.DataFrame, 
    fps: float, 
    image_width: int, 
    image_height: int,
    hand_label: str,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates key kinematic metrics and a dexterity score from a DataFrame of landmark data for a single hand.
    This is the single source of truth for all kinematic calculations.
    """
    # Ensure we're working with a fresh copy
    hand_df = hand_df.copy()

    # Calculate a single point to represent the hand's position (center of palm)
    # Using landmarks 0 (wrist), 5 (index MCP), 9 (middle MCP), 13 (ring MCP), 17 (pinky MCP)
    palm_landmarks = [0, 5, 9, 13, 17]
    hand_df['palm_x'] = hand_df.groupby('time')['x'].transform(lambda g: g[hand_df['landmark_id'].isin(palm_landmarks)].mean())
    hand_df['palm_y'] = hand_df.groupby('time')['y'].transform(lambda g: g[hand_df['landmark_id'].isin(palm_landmarks)].mean())

    # We only need one position per timestamp, so we drop duplicates
    pos_df = hand_df[['time', 'palm_x', 'palm_y']].drop_duplicates().sort_values('time').reset_index(drop=True)

    if len(pos_df) < 4 or fps == 0:
        return {"time_series": {}, "summary": {}}

    dt = 1.0 / fps
    pos = pos_df[['palm_x', 'palm_y']].to_numpy() * [image_width, image_height]
    timestamps = pos_df['time'].to_numpy()

    # Calculate derivatives
    velocity_vec = np.diff(pos, axis=0) / dt
    velocity_mag = np.linalg.norm(velocity_vec, axis=1)

    acceleration_vec = np.diff(velocity_vec, axis=0) / dt
    acceleration_mag = np.linalg.norm(acceleration_vec, axis=1)

    jerk_vec = np.diff(acceleration_vec, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk_vec, axis=1)
    
    # Detect and remove outliers from jerk data (if enabled)
    if params and params.get('outlier_detection_enabled', True):
        method = params.get('outlier_detection_method', 'iqr')
        threshold = params.get('outlier_threshold', 1.30)
        jerk_cleaned, jerk_outlier_mask = detect_and_remove_outliers(jerk_mag, method=method, threshold=threshold)
    else:
        jerk_cleaned = jerk_mag
        jerk_outlier_mask = np.array([False] * len(jerk_mag))
    
    # Log outlier information
    outliers_removed = np.sum(jerk_outlier_mask)
    if outliers_removed > 0:
        logger.info(f"Removed {outliers_removed} jerk outliers from {len(jerk_mag)} data points for {hand_label} hand")
    
    # Use cleaned jerk data for further calculations
    jerk_for_calculations = jerk_cleaned if len(jerk_cleaned) > 0 else jerk_mag
    
    # Calculate cumulative jerk (integral of jerk magnitude over time)
    cumulative_jerk = np.cumsum(jerk_for_calculations) * dt
    
    # Calculate cumulative path length over time
    path_segments = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cumulative_path = np.cumsum(path_segments)

    # Calculate cumulative metrics
    total_path_length = cumulative_path[-1] if cumulative_path.size > 0 else 0
    path_length_for_jerk = np.linalg.norm(pos[-1] - pos[0])
    movement_duration = timestamps[-1] - timestamps[0]

    # Calculate dimensionless jerk and dexterity score using cleaned jerk data
    log_dimensionless_jerk = 0
    dimensionless_jerk_teulings = 0
    squared_jerk_integral = 0
    if path_length_for_jerk > 1e-6 and movement_duration > 1e-6 and jerk_for_calculations.size > 0:
        squared_jerk_integral = np.sum(jerk_for_calculations**2) * dt
        # Teulings' dimensionless jerk formula
        dimensionless_jerk_teulings = ((movement_duration**3) / (path_length_for_jerk**2)) * squared_jerk_integral
        log_dimensionless_jerk = -np.log(dimensionless_jerk_teulings + 1e-9)
    
    dexterity_score = max(0, 100 + (log_dimensionless_jerk * 10))

    # Calculate average metrics (new requirements)
    average_velocity = np.mean(velocity_mag) if velocity_mag.size > 0 else 0
    average_acceleration = np.mean(acceleration_mag) if acceleration_mag.size > 0 else 0

    # Prepare time-series data for plotting
    def create_time_series(data, label):
        # Ensure data and timestamps align, accounting for np.diff shortening the array
        ts = timestamps[len(timestamps) - len(data):]
        return [{"time": t, label: v, "hand": hand_label} for t, v in zip(ts, data)]
    
    # Create time series for jerk without outliers for plotting
    # We need to map the cleaned jerk data back to timestamps, excluding outlier points
    jerk_times = timestamps[3:]  # jerk starts at index 3 due to triple differentiation
    jerk_time_series = []
    if len(jerk_for_calculations) > 0:
        # If outliers were removed, we need to handle the timestamp mapping carefully
        if outliers_removed > 0:
            # Create time series only for non-outlier points
            valid_indices = ~jerk_outlier_mask
            valid_times = jerk_times[valid_indices]
            jerk_time_series = [{"time": t, "Jerk": v, "hand": hand_label} for t, v in zip(valid_times, jerk_for_calculations)]
        else:
            # No outliers removed, use all data
            jerk_time_series = [{"time": t, "Jerk": v, "hand": hand_label} for t, v in zip(jerk_times, jerk_for_calculations)]

    results = {
        "time_series": {
            "velocity": create_time_series(velocity_mag, "Velocity"),
            "acceleration": create_time_series(acceleration_mag, "Acceleration"),
            "jerk": jerk_time_series,
            "cumulative_jerk": create_time_series(cumulative_jerk, "Cumulative Jerk"),
            "cumulative_path_length": create_time_series(cumulative_path, "Cumulative Path Length"),
        },
        "summary": {
            "total_path_pixels": round(total_path_length, 2),
            "peak_velocity_pps": round(np.max(velocity_mag) if velocity_mag.size > 0 else 0, 2),
            "peak_acceleration_pps2": round(np.max(acceleration_mag) if acceleration_mag.size > 0 else 0, 2),
            "peak_jerk_pps3": round(np.max(jerk_for_calculations) if jerk_for_calculations.size > 0 else 0, 2),
            "total_cumulative_jerk": round(cumulative_jerk[-1] if cumulative_jerk.size > 0 else 0, 2),
            "log_dimensionless_jerk": round(log_dimensionless_jerk, 2),
            "dexterity_score": round(dexterity_score, 2),
            "outliers_removed": int(outliers_removed),
            # New metrics as requested
            "average_velocity_pps": round(average_velocity, 2),
            "average_acceleration_pps2": round(average_acceleration, 2),
            "dimensionless_jerk": round(dimensionless_jerk_teulings, 6),
            "squared_jerk_integral": round(squared_jerk_integral, 6),
        }
    }
    return results

def perform_analysis_on_dataframe(analysis_id: str, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs kinematic analysis on a pandas DataFrame of hand landmarks.
    This is the core calculation logic that splits data by hand and aggregates the results.
    """
    if df.empty:
        return {"summary": {}, "analysis": {}}

    image_width = params.get("image_width", 1)
    image_height = params.get("image_height", 1)
    fps = params.get("fps", 30)

    # Aggregate all time-series and summary data
    all_time_series = {}
    all_summaries = {}

    for hand_label in ["Left", "Right"]:
        hand_df = df[df['hand'] == hand_label].copy()
        
        if len(hand_df) > 4:
            # The actual calculation is now consolidated in this function
            hand_results = calculate_kinematics_and_metrics(hand_df, fps, image_width, image_height, hand_label, params)
            
            # Store summary and merge time-series data
            all_summaries[hand_label] = hand_results.get("summary", {})
            for key, value in hand_results.get("time_series", {}).items():
                if key not in all_time_series:
                    all_time_series[key] = []
                all_time_series[key].extend(value)
    
    # Calculate overall dexterity score
    left_score = all_summaries.get("Left", {}).get("dexterity_score", 0)
    right_score = all_summaries.get("Right", {}).get("dexterity_score", 0)
    overall_dexterity = (left_score + right_score) / 2 if left_score > 0 and right_score > 0 else max(left_score, right_score)
    
    all_summaries["Overall"] = {"dexterity_score": round(overall_dexterity, 2)}
    
    return {"summary": all_summaries, "analysis": all_time_series}

def perform_analysis(analysis_id: str, file_path: str, params: Dict[str, Any], loop: asyncio.AbstractEventLoop):
    try:
        logger.info(f"Starting analysis for {analysis_id} with params: {params}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=params.get('min_detection_confidence', 0.75),
            min_tracking_confidence=params.get('min_tracking_confidence', 0.75)
        )
        
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file {file_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logger.warning(f"Video FPS is 0 for {file_path}. Defaulting to 30 FPS.")
                fps = 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            params.update({"fps": fps, "image_width": frame_width, "image_height": frame_height})
            db_execute("UPDATE analyses SET analysis_parameters = ? WHERE id = ?", (json.dumps(params), analysis_id))

            # Set up video writer for annotated output. Prefer H.264 (avc1) for broad browser compatibility; fallback to mp4v.
            annotated_video_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (frame_width, frame_height))

            # Fallback to MPEG-4 Part 2 if H.264 isn't available in the environment
            if not out.isOpened():
                logger.warning("H.264 (avc1) encoder not available, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (frame_width, frame_height))

            # --- Data Collection ---
            filters: Dict[str, Dict[int, Tuple[OneEuroFilter, ...]]] = {"Left": {}, "Right": {}}
            raw_landmarks_data = []
            filtered_landmarks_data = []

            last_sent_ts = 0.0  # throttle websocket to ~10 fps
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_s = timestamp_ms / 1000.0
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                annotated_frame = frame.copy()

                if detection_result.hand_landmarks:
                    for i, hand_landmarks_list in enumerate(detection_result.hand_landmarks):
                        handedness = detection_result.handedness[i][0].category_name
                        if handedness not in filters: continue

                        # Store raw landmarks before filtering
                        for lm_idx, lm in enumerate(hand_landmarks_list):
                            raw_landmarks_data.append({'time': timestamp_s, 'hand': handedness, 'landmark_id': lm_idx, 'x': lm.x, 'y': lm.y, 'z': lm.z})

                        # Filter landmarks for analysis with adaptive parameters
                        for lm_idx, lm in enumerate(hand_landmarks_list):
                            if lm_idx not in filters[handedness]:
                                # Use configurable filtering based on landmark type and user settings
                                # Get user-defined multipliers
                                fingertip_mult = params.get('fingertip_filter_multiplier', 1.0)
                                joint_mult = params.get('joint_filter_multiplier', 1.0)
                                palm_mult = params.get('palm_responsiveness_multiplier', 1.0)
                                
                                if lm_idx in [4, 8, 12, 16, 20]:  # Fingertips - configurable strictness
                                    strictness_factor = 0.1 * fingertip_mult  # Base strictness adjusted by user preference
                                    min_cutoff = params.get('filter_min_cutoff', 0.003) * strictness_factor
                                    beta = params.get('filter_beta', 0.2) * (0.2 * fingertip_mult)
                                elif lm_idx in [0, 1, 5, 9, 13, 17]:  # Wrist and palm base landmarks - configurable responsiveness
                                    responsiveness_factor = 1.5 * palm_mult  # Base responsiveness adjusted by user preference
                                    min_cutoff = params.get('filter_min_cutoff', 0.003) * responsiveness_factor
                                    beta = params.get('filter_beta', 0.2) * (1.3 * palm_mult)
                                elif lm_idx in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]:  # Finger joints - configurable
                                    joint_factor = 0.3 * joint_mult  # Base joint filtering adjusted by user preference
                                    min_cutoff = params.get('filter_min_cutoff', 0.003) * joint_factor
                                    beta = params.get('filter_beta', 0.2) * (0.5 * joint_mult)
                                else:  # Other landmarks - use base settings
                                    min_cutoff = params.get('filter_min_cutoff', 0.003)
                                    beta = params.get('filter_beta', 0.2)
                                
                                filters[handedness][lm_idx] = (
                                    OneEuroFilter(t0=timestamp_s, x0=lm.x, min_cutoff=min_cutoff, beta=beta),
                                    OneEuroFilter(t0=timestamp_s, x0=lm.y, min_cutoff=min_cutoff, beta=beta),
                                    OneEuroFilter(t0=timestamp_s, x0=lm.z, min_cutoff=min_cutoff, beta=beta)
                                )
                            fx, fy, fz = filters[handedness][lm_idx]
                            filtered_x, filtered_y, filtered_z = fx(timestamp_s, lm.x), fy(timestamp_s, lm.y), fz(timestamp_s, lm.z)
                            filtered_landmarks_data.append({'time': timestamp_s, 'hand': handedness, 'landmark_id': lm_idx, 'x': filtered_x, 'y': filtered_y, 'z': filtered_z})
                        
                        # Draw landmarks and connections on the frame (using original landmarks for stable drawing)
                        proto_landmarks = landmark_pb2.NormalizedLandmarkList()
                        proto_landmarks.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks_list])
                        mp.solutions.drawing_utils.draw_landmarks(
                            annotated_frame, proto_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(), 
                            mp.solutions.drawing_styles.get_default_hand_connections_style())

                # Write the annotated frame to the output video
                out.write(annotated_frame)

                # Send frame to WebSocket for live viewing (throttled)
                try:
                    now = time.time()
                    if now - last_sent_ts >= 0.1:  # ~10 fps
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        asyncio.run_coroutine_threadsafe(manager.send_frame(analysis_id, buffer.tobytes()), loop)
                        last_sent_ts = now
                except Exception as e:
                    logger.warning(f"Could not send frame to websocket for {analysis_id}: {e}")
            
            cap.release()
            out.release()  # Close the annotated video writer
            
            # --- Data Persistence and Analysis ---
            if not filtered_landmarks_data:
                logger.warning(f"No landmarks detected for analysis {analysis_id}. Finishing as failed.")
                db_execute("UPDATE analyses SET status='failed', error_message='No landmarks detected in video.' WHERE id=?", (analysis_id,))
                return

            # Create DataFrames
            original_df = pd.DataFrame(raw_landmarks_data)
            filtered_df = pd.DataFrame(filtered_landmarks_data)

            # Store both original (unfiltered) and working (filtered) data
            original_json = original_df.to_json(orient='records')
            filtered_json = filtered_df.to_json(orient='records')
            db_execute("UPDATE analyses SET raw_landmarks_json = ?, original_landmarks_json = ?, annotated_video_path = ? WHERE id = ?", 
                      (filtered_json, original_json, annotated_video_path, analysis_id))
            logger.info(f"Stored raw (filtered) and original (unfiltered) landmark data for {analysis_id}")
            logger.info(f"Saved annotated video to {annotated_video_path}")

            # Perform the actual kinematic analysis on the filtered data
            analysis_results = perform_analysis_on_dataframe(analysis_id, filtered_df, params)

            # Get created_at and original_name for the final object
            analysis_record = db_query_one("SELECT created_at, original_name FROM analyses WHERE id = ?", (analysis_id,))

            # Structure the final JSON output that will be stored
            output = {
                "id": analysis_id,
                "original_name": analysis_record['original_name'] if analysis_record else 'N/A',
                "status": "completed",
                "created_at": analysis_record['created_at'] if analysis_record else datetime.utcnow().isoformat(),
                "analysis_parameters": params,
                "summary": analysis_results.get("summary", {}),
                "analysis": analysis_results.get("analysis", {}),
                "video_path": f"/api/analysis/{analysis_id}/video",
                "annotated_video_path": f"/api/analysis/{analysis_id}/annotated_video"
            }

            db_execute("UPDATE analyses SET status='completed', results=? WHERE id=?", (json.dumps(output), analysis_id))
            logger.info(f"Successfully completed analysis for {analysis_id}")

    except Exception as e:
        logger.error(f"Analysis failed for {analysis_id}: {e}", exc_info=True)
        db_execute("UPDATE analyses SET status='failed', error_message=? WHERE id=?", (str(e), analysis_id))
    finally:
        # Ensure the websocket connection is closed if it's still open
        try:
            asyncio.run_coroutine_threadsafe(manager.close_connection(analysis_id), loop)
        except Exception as e:
            logger.warning(f"Could not close websocket for {analysis_id}: {e}")

# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
# Restrict CORS to allowed origins (comma-separated). If not provided, include localhost, 127.0.0.1 and local LAN IP on port 3000.
_env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
if _env_origins == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # credentials not allowed with wildcard origins
        allow_methods=["*"],
        allow_headers=["*"]
    )
else:
    origins = {o.strip() for o in _env_origins.split(",") if o.strip()}
    if not origins:
        # default dev origins
        origins.update({
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        })
        # Try to detect LAN IP and allow it
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
            if lan_ip and not lan_ip.startswith("127."):
                origins.add(f"http://{lan_ip}:3000")
        except Exception:
            pass
    # Build a permissive regex for localhost/127.0.0.1 on any port, plus LAN IP on any port
    origin_patterns = [r"^http://localhost(:\\d+)?$", r"^http://127\\.0\\.0\\.1(:\\d+)?$"]
    try:
        lan_pat = None
        if 'lan_ip' in locals() and lan_ip:
            escaped_lan_ip = lan_ip.replace('.', r'\.')
            lan_pat = rf"^http://{escaped_lan_ip}(:\d+)?$"
        if lan_pat:
            origin_patterns.append(lan_pat)
    except Exception:
        pass
    allow_origin_regex = "|".join(origin_patterns)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_origin_regex=allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

@app.get("/health")
async def health_check(): return {"status": "ok"}

# --- API Endpoints ---
@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    min_detection_confidence: float = Form(0.75), 
    min_tracking_confidence: float = Form(0.75), 
    filter_min_cutoff: float = Form(0.003), 
    filter_beta: float = Form(0.2),
    # Outlier detection parameters
    outlier_detection_enabled: bool = Form(True),
    outlier_detection_method: str = Form('iqr'),
    outlier_threshold: float = Form(1.30),
    # Advanced filtering parameters
    fingertip_filter_multiplier: float = Form(1.0),
    joint_filter_multiplier: float = Form(1.0),
    palm_responsiveness_multiplier: float = Form(1.0)
):
    analysis_id = str(uuid.uuid4())
    # Sanitize filename and validate extension server-side
    original_filename = file.filename or "uploaded_video"
    safe_name = "".join([c for c in original_filename if c.isalnum() or c in "._- "]).strip().replace(" ", "_")
    name_root, name_ext = os.path.splitext(safe_name)
    allowed_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    if name_ext.lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{name_ext}'. Allowed: {', '.join(sorted(allowed_exts))}")
    safe_name = f"{name_root[:100]}{name_ext.lower()}" if name_root else f"video{name_ext.lower()}"
    file_path = os.path.join(UPLOAD_DIR, f"{analysis_id}_{safe_name}")
    loop = asyncio.get_running_loop()
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Enforce max file size (2GB)
        max_size_bytes = int(float(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))))
        try:
            saved_size = os.path.getsize(file_path)
            if saved_size > max_size_bytes:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="File too large. Max 2GB")
        except OSError:
            pass
        analysis_params = {
            "min_detection_confidence": min_detection_confidence, "min_tracking_confidence": min_tracking_confidence,
            "filter_min_cutoff": filter_min_cutoff, "filter_beta": filter_beta,
            # Outlier detection parameters
            "outlier_detection_enabled": outlier_detection_enabled,
            "outlier_detection_method": outlier_detection_method,
            "outlier_threshold": outlier_threshold,
            # Advanced filtering parameters
            "fingertip_filter_multiplier": fingertip_filter_multiplier,
            "joint_filter_multiplier": joint_filter_multiplier,
            "palm_responsiveness_multiplier": palm_responsiveness_multiplier}
        db_execute("INSERT INTO analyses (id, original_name, status, analysis_parameters, video_path) VALUES (?, ?, ?, ?, ?)",
            (analysis_id, safe_name, 'processing', json.dumps(analysis_params), file_path))
        background_tasks.add_task(perform_analysis, analysis_id, file_path, analysis_params, loop)
        return JSONResponse(status_code=200, content={"analysis_id": analysis_id, "message": "Analysis started"})
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"There was an error processing the file: {e}"})

@app.get("/api/analyses")
async def get_analyses():
    analyses = db_query_all("SELECT id, original_name, status, created_at, error_message, results, analysis_parameters FROM analyses ORDER BY created_at DESC")
    return [dict(row) for row in analyses] if analyses else []

@app.get("/api/analyses/download_raw_landmarks_csv")
async def download_all_raw_landmarks_csv():
    analyses = db_query_all("SELECT id, original_name, raw_landmarks_json, analysis_parameters FROM analyses WHERE status = 'completed' ORDER BY created_at DESC")
    if not analyses:
        raise HTTPException(status_code=404, detail="No completed analyses found to export.")
    output = io.StringIO()
    writer = csv.writer(output)
    headers = [
        'analysis_id', 'original_name',
        'timestamp', 'hand', 'landmark_id',
        'x_normalized', 'y_normalized', 'z_normalized',
        'x_pixels', 'y_pixels'
    ]
    writer.writerow(headers)
    for row in analyses:
        try:
            raw_json = row['raw_landmarks_json']
            if not raw_json:
                continue
            params = json.loads(row['analysis_parameters']) if row['analysis_parameters'] else {}
            image_width = params.get('image_width', 1)
            image_height = params.get('image_height', 1)
            df = pd.read_json(raw_json)
            if df.empty:
                continue
            df['x_pixels'] = df['x'] * image_width
            df['y_pixels'] = df['y'] * image_height
            for _, r in df.iterrows():
                writer.writerow([
                    row['id'], row['original_name'],
                    r['time'], r['hand'], r['landmark_id'],
                    r['x'], r['y'], r['z'],
                    r['x_pixels'], r['y_pixels']
                ])
        except Exception:
            continue
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=all_raw_landmarks_{time.strftime('%Y%m%d')}.csv"})

@app.get("/api/analyses/download_timeseries_csv")
async def download_all_timeseries_csv():
    analyses = db_query_all("SELECT id, original_name, results FROM analyses WHERE status = 'completed' ORDER BY created_at DESC")
    if not analyses:
        raise HTTPException(status_code=404, detail="No completed analyses found to export.")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['analysis_id', 'original_name', 'timestamp_seconds', 'hand', 'velocity_pps', 'acceleration_pps2', 'jerk_pps3', 'cumulative_jerk_pps3s', 'cumulative_path_length_pixels'])
    for a in analyses:
        try:
            results = json.loads(a['results']) if a['results'] else {}
            analysis_data = results.get('analysis', {})
            all_timestamps = set()
            for metric_data in analysis_data.values():
                for item in metric_data:
                    all_timestamps.add(item['time'])
            sorted_times = sorted(all_timestamps)
            velocity_lookup = {(item['time'], item['hand']): item.get('Velocity') for item in analysis_data.get('velocity', [])}
            acceleration_lookup = {(item['time'], item['hand']): item.get('Acceleration') for item in analysis_data.get('acceleration', [])}
            jerk_lookup = {(item['time'], item['hand']): item.get('Jerk') for item in analysis_data.get('jerk', [])}
            cumulative_jerk_lookup = {(item['time'], item['hand']): item.get('Cumulative Jerk') for item in analysis_data.get('cumulative_jerk', [])}
            path_lookup = {(item['time'], item['hand']): item.get('Cumulative Path Length') for item in analysis_data.get('cumulative_path_length', [])}
            for hand_label in ['Left', 'Right']:
                for t in sorted_times:
                    key = (t, hand_label)
                    if key in velocity_lookup or key in acceleration_lookup or key in jerk_lookup or key in cumulative_jerk_lookup or key in path_lookup:
                        writer.writerow([
                            a['id'], a['original_name'],
                            t, hand_label,
                            velocity_lookup.get(key, ''),
                            acceleration_lookup.get(key, ''),
                            jerk_lookup.get(key, ''),
                            cumulative_jerk_lookup.get(key, ''),
                            path_lookup.get(key, '')
                        ])
        except Exception:
            continue
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=all_timeseries_{time.strftime('%Y%m%d')}.csv"})

@app.get("/api/analyses/download_final_values_csv")
async def download_all_final_values_csv():
    # Reuse existing aggregated metrics endpoint behavior for compatibility
    return await download_all_metrics_csv()
@app.get("/api/analyses/download_metrics_csv")
async def download_all_metrics_csv():
    analyses = db_query_all("SELECT id, original_name, created_at, results FROM analyses WHERE status = 'completed' ORDER BY created_at DESC")
    if not analyses: raise HTTPException(status_code=404, detail="No completed analyses found to export.")
    output = io.StringIO()
    headers = [
        'analysis_id', 'original_name', 'analysis_date',
        'left_total_path_pixels', 'left_peak_velocity_pps', 'left_average_velocity_pps', 'left_peak_acceleration_pps2', 'left_average_acceleration_pps2', 'left_peak_jerk_pps3', 'left_log_dimensionless_jerk', 'left_dimensionless_jerk', 'left_squared_jerk_integral',
        'right_total_path_pixels', 'right_peak_velocity_pps', 'right_average_velocity_pps', 'right_peak_acceleration_pps2', 'right_average_acceleration_pps2', 'right_peak_jerk_pps3', 'right_log_dimensionless_jerk', 'right_dimensionless_jerk', 'right_squared_jerk_integral']
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for analysis in analyses:
        try:
            results = json.loads(analysis['results']) if analysis['results'] else {}
            summary = results.get('summary', {})
            row = {'analysis_id': analysis['id'], 'original_name': analysis['original_name'], 'analysis_date': analysis['created_at']}
            for hand in ['left', 'right']:
                hand_summary = summary.get(hand.capitalize(), {})
                for key, value in hand_summary.items():
                    csv_key = f"{hand}_{key}"
                    if csv_key in headers: row[csv_key] = value
            writer.writerow(row)
        except (json.JSONDecodeError, TypeError): continue
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=all_analysis_metrics_{time.strftime('%Y%m%d')}.csv"})

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_details(analysis_id: str):
    analysis = db_query_one("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis: raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_dict = dict(analysis)
    if analysis_dict.get('results'):
        try:
            # The 'results' column contains the full JSON object we want to return
            return JSONResponse(content=json.loads(analysis_dict['results']))
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, return the raw record but log an error
            logger.error(f"Failed to parse 'results' JSON for analysis {analysis_id}")
            # Fallback to returning the raw dictionary, but this might not work with the frontend
            return JSONResponse(content=analysis_dict)
    
    return JSONResponse(content=analysis_dict)

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    analysis = db_query_one("SELECT video_path, annotated_video_path FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis: raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Delete original video file
    video_path = analysis['video_path']
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            logger.info(f"Deleted original video file: {video_path}")
        except Exception as e:
            logger.error(f"Error deleting original video file {video_path}: {e}")
    
    # Delete annotated video file
    annotated_video_path = analysis['annotated_video_path']
    if annotated_video_path and os.path.exists(annotated_video_path):
        try:
            os.remove(annotated_video_path)
            logger.info(f"Deleted annotated video file: {annotated_video_path}")
        except Exception as e:
            logger.error(f"Error deleting annotated video file {annotated_video_path}: {e}")
    
    db_execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    return {"message": "Analysis deleted successfully"}

@app.get("/api/analysis/{analysis_id}/video")
async def get_analysis_video(request: Request, analysis_id: str):
    analysis = db_query_one("SELECT annotated_video_path FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['annotated_video_path'] or not os.path.exists(analysis['annotated_video_path']):
        raise HTTPException(status_code=404, detail="Annotated video file not found for this analysis.")
    
    video_path = analysis['annotated_video_path']

    return await _range_stream_video(request, video_path)

@app.get("/api/analysis/{analysis_id}/original_video")
async def get_original_video(request: Request, analysis_id: str):
    analysis = db_query_one("SELECT video_path FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['video_path'] or not os.path.exists(analysis['video_path']):
        raise HTTPException(status_code=404, detail="Original video file not found for this analysis.")
    
    video_path = analysis['video_path']
    return await _range_stream_video(request, video_path)

@app.get("/api/analysis/{analysis_id}/annotated_video")
async def get_annotated_video(request: Request, analysis_id: str):
    analysis = db_query_one("SELECT annotated_video_path FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['annotated_video_path'] or not os.path.exists(analysis['annotated_video_path']):
        raise HTTPException(status_code=404, detail="Annotated video file not found for this analysis.")
    
    video_path = analysis['annotated_video_path']
    return await _range_stream_video(request, video_path)

def _iter_file_range(path: str, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(path, 'rb') as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            yield data
            remaining -= len(data)

async def _range_stream_video(request: Request, path: str):
    try:
        file_size = os.path.getsize(path)
    except Exception as e:
        logger.error(f"Cannot stat video file {path}: {e}")
        raise HTTPException(status_code=500, detail="Video file is inaccessible.")

    range_header = request.headers.get('range')
    if range_header:
        # Example: bytes=0-1023
        try:
            units, _, range_spec = range_header.partition('=')
            if units != 'bytes':
                raise ValueError("Only 'bytes' ranges are supported")
            start_str, _, end_str = range_spec.partition('-')
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            if start > end or end >= file_size:
                raise ValueError("Invalid range")
        except Exception:
            # Malformed range
            return JSONResponse(status_code=416, content={"detail": "Invalid Range header"})

        headers = {
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "public, max-age=3600",
        }
        return StreamingResponse(_iter_file_range(path, start, end), status_code=206, headers=headers)

    # No Range header: return full file
    headers = {
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Cache-Control": "public, max-age=3600",
    }
    return StreamingResponse(_iter_file_range(path, 0, file_size - 1), status_code=200, headers=headers)

@app.get("/api/analysis/{analysis_id}/download_csv")
async def download_analysis_csv(analysis_id: str):
    analysis = db_query_one("SELECT raw_landmarks_json, original_name, results, analysis_parameters FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['raw_landmarks_json']:
        raise HTTPException(status_code=404, detail="Analysis data not found or is incomplete. Please re-analyze the video.")
    
    try:
        # Parse the data
        raw_landmarks_df = pd.read_json(analysis['raw_landmarks_json'])
        results = json.loads(analysis['results']) if analysis['results'] else {}
        params = json.loads(analysis['analysis_parameters']) if analysis['analysis_parameters'] else {}
        
        # Get FPS and image dimensions
        fps = params.get('fps', 30)
        image_width = params.get('image_width', 1)
        image_height = params.get('image_height', 1)
        
        # Get summary data for final values
        summary = results.get('summary', {})
        left_summary = summary.get('Left', {})
        right_summary = summary.get('Right', {})
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers with new final value columns
        headers = [
            'timestamp', 'hand', 'landmark_id', 
            'x_normalized', 'y_normalized', 'z_normalized',
            'x_pixels', 'y_pixels',
            'palm_center_x_pixels', 'palm_center_y_pixels',
            'velocity_pps', 'acceleration_pps2', 'jerk_pps3',
            'cumulative_path_length_pixels',
            # Final values columns
            'final_cumulative_jerk_pps3', 
            'final_dimensionless_jerk',
            'final_log_dimensionless_jerk',
            'final_total_path_length_pixels',
            'final_peak_velocity_pps',
            'final_average_velocity_pps',  # NEW
            'final_peak_acceleration_pps2',
            'final_average_acceleration_pps2',  # NEW
            'final_peak_jerk_pps3',
            'final_squared_jerk_integral'  # NEW
        ]
        writer.writerow(headers)
        
        # Process each hand separately
        for hand_label in ['Left', 'Right']:
            hand_df = raw_landmarks_df[raw_landmarks_df['hand'] == hand_label].copy()
            
            if hand_df.empty:
                continue
                
            # Get final values for this hand
            hand_summary = left_summary if hand_label == 'Left' else right_summary
            final_total_path = hand_summary.get('total_path_pixels', '')
            final_peak_velocity = hand_summary.get('peak_velocity_pps', '')
            final_peak_acceleration = hand_summary.get('peak_acceleration_pps2', '')
            final_peak_jerk = hand_summary.get('peak_jerk_pps3', '')
            final_log_dimensionless_jerk = hand_summary.get('log_dimensionless_jerk', '')
                
            # Convert normalized coordinates to pixels
            hand_df['x_pixels'] = hand_df['x'] * image_width
            hand_df['y_pixels'] = hand_df['y'] * image_height
            
            # Calculate palm center for each timestamp
            palm_landmarks = [0, 5, 9, 13, 17]
            palm_data = []
            
            for time_val in hand_df['time'].unique():
                time_df = hand_df[hand_df['time'] == time_val]
                palm_df = time_df[time_df['landmark_id'].isin(palm_landmarks)]
                if not palm_df.empty:
                    palm_x = palm_df['x_pixels'].mean()
                    palm_y = palm_df['y_pixels'].mean()
                    palm_data.append({
                        'time': time_val,
                        'palm_x': palm_x,
                        'palm_y': palm_y
                    })
            
            palm_df = pd.DataFrame(palm_data)
            
            # Calculate kinematics on palm center
            velocity_data = {}
            acceleration_data = {}
            jerk_data = {}
            cumulative_path_data = {}
            cumulative_jerk_data = {}
            
            if len(palm_df) >= 4:
                dt = 1.0 / fps
                pos = palm_df[['palm_x', 'palm_y']].to_numpy()
                timestamps = palm_df['time'].to_numpy()
                
                # Velocity
                velocity_vec = np.diff(pos, axis=0) / dt
                velocity_mag = np.linalg.norm(velocity_vec, axis=1)
                for i, t in enumerate(timestamps[1:]):
                    velocity_data[t] = velocity_mag[i]
                
                # Acceleration
                if len(velocity_vec) > 1:
                    acceleration_vec = np.diff(velocity_vec, axis=0) / dt
                    acceleration_mag = np.linalg.norm(acceleration_vec, axis=1)
                    for i, t in enumerate(timestamps[2:]):
                        acceleration_data[t] = acceleration_mag[i]
                
                # Jerk and cumulative jerk
                if len(velocity_vec) > 2:
                    jerk_vec = np.diff(acceleration_vec, axis=0) / dt
                    jerk_mag = np.linalg.norm(jerk_vec, axis=1)
                    cumulative_jerk = np.cumsum(jerk_mag) * dt  # Integrate jerk over time
                    for i, t in enumerate(timestamps[3:]):
                        jerk_data[t] = jerk_mag[i]
                        cumulative_jerk_data[t] = cumulative_jerk[i]
                
                # Cumulative path length
                path_segments = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                cumulative_path = np.cumsum(path_segments)
                cumulative_path_data[timestamps[0]] = 0  # First point has 0 path
                for i, t in enumerate(timestamps[1:]):
                    cumulative_path_data[t] = cumulative_path[i]
            
            # Calculate final cumulative jerk and dimensionless jerk if not in summary
            final_cumulative_jerk = ''
            final_dimensionless_jerk = ''
            final_average_velocity_pps = ''
            final_average_acceleration_pps2 = ''
            final_squared_jerk_integral = ''
            final_peak_jerk_pps3 = final_peak_jerk  # backward compatible variable name
            if cumulative_jerk_data:
                final_cumulative_jerk = max(cumulative_jerk_data.values())
                # Calculate dimensionless jerk if we have the data
                if len(palm_df) >= 4 and final_total_path and final_total_path != '':
                    movement_duration = timestamps[-1] - timestamps[0]
                    path_length_for_jerk = np.linalg.norm(pos[-1] - pos[0])
                    if path_length_for_jerk > 1e-6 and movement_duration > 1e-6:
                        squared_jerk_integral_val = np.sum(jerk_mag**2) * dt
                        final_dimensionless_jerk = ((movement_duration**3) / (path_length_for_jerk**2)) * squared_jerk_integral_val
                        final_squared_jerk_integral = squared_jerk_integral_val

                # Compute averages
                if len(velocity_vec) > 0:
                    final_average_velocity_pps = float(np.mean(velocity_mag))
                if 'acceleration_mag' in locals() and len(acceleration_mag) > 0:
                    final_average_acceleration_pps2 = float(np.mean(acceleration_mag))
            
            # Merge palm center data back to hand_df
            palm_center_map = {row['time']: (row['palm_x'], row['palm_y']) 
                              for _, row in palm_df.iterrows()}
            
            # Write rows
            for _, row in hand_df.iterrows():
                time_val = row['time']
                palm_x, palm_y = palm_center_map.get(time_val, (None, None))
                
                writer.writerow([
                    row['time'],
                    row['hand'],
                    row['landmark_id'],
                    row['x'],
                    row['y'],
                    row['z'],
                    row['x_pixels'],
                    row['y_pixels'],
                    palm_x,
                    palm_y,
                    velocity_data.get(time_val, ''),
                    acceleration_data.get(time_val, ''),
                    jerk_data.get(time_val, ''),
                    cumulative_path_data.get(time_val, ''),
                    # Final values (same for all rows of this hand)
                    cumulative_jerk_data.get(time_val, ''),  # Current cumulative jerk
                    final_dimensionless_jerk,
                    final_log_dimensionless_jerk,
                    final_total_path,
                    final_peak_velocity,
                    final_average_velocity_pps,  # NEW
                    final_peak_acceleration,
                    final_average_acceleration_pps2,  # NEW
                    final_peak_jerk_pps3,
                    final_squared_jerk_integral  # NEW
                ])
        
        output.seek(0)
        clean_filename = "".join([c for c in analysis['original_name'] if c.isalpha() or c.isdigit() or c in '._-']).rstrip()
        return StreamingResponse(
            output, 
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=kinematic_data_{clean_filename}_{analysis_id[:8]}.csv"}
        )
        
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to generate CSV for analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate CSV data.")

@app.get("/api/analysis/{analysis_id}/download_timeseries_csv")
async def download_analysis_timeseries_csv(analysis_id: str):
    """Download time-series kinematic data (velocity, acceleration, jerk) as CSV"""
    analysis = db_query_one("SELECT results, original_name FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['results']:
        raise HTTPException(status_code=404, detail="Analysis results not found.")
    
    try:
        results = json.loads(analysis['results'])
        analysis_data = results.get('analysis', {})
        
        # Prepare CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'timestamp_seconds', 'hand',
            'velocity_pps', 'acceleration_pps2', 'jerk_pps3',
            'cumulative_jerk_pps3s', 'cumulative_path_length_pixels'
        ])
        
        # Get all unique timestamps across all metrics
        all_timestamps = set()
        for metric_data in analysis_data.values():
            for item in metric_data:
                all_timestamps.add(item['time'])
        
        # Sort timestamps
        sorted_times = sorted(all_timestamps)
        
        # Create lookup dictionaries for each metric and hand
        velocity_lookup = {(item['time'], item['hand']): item['Velocity'] 
                          for item in analysis_data.get('velocity', [])}
        acceleration_lookup = {(item['time'], item['hand']): item['Acceleration'] 
                              for item in analysis_data.get('acceleration', [])}
        jerk_lookup = {(item['time'], item['hand']): item['Jerk'] 
                      for item in analysis_data.get('jerk', [])}
        cumulative_jerk_lookup = {(item['time'], item['hand']): item['Cumulative Jerk'] 
                                 for item in analysis_data.get('cumulative_jerk', [])}
        path_lookup = {(item['time'], item['hand']): item['Cumulative Path Length'] 
                      for item in analysis_data.get('cumulative_path_length', [])}
        
        # Write data for each hand
        for hand in ['Left', 'Right']:
            for t in sorted_times:
                # Check if this hand has any data at this timestamp
                key = (t, hand)
                if key in velocity_lookup or key in acceleration_lookup or key in jerk_lookup or key in cumulative_jerk_lookup or key in path_lookup:
                    writer.writerow([
                        t,
                        hand,
                        velocity_lookup.get(key, ''),
                        acceleration_lookup.get(key, ''),
                        jerk_lookup.get(key, ''),
                        cumulative_jerk_lookup.get(key, ''),
                        path_lookup.get(key, '')
                    ])
        
        output.seek(0)
        clean_filename = "".join([c for c in analysis['original_name'] if c.isalpha() or c.isdigit() or c in '._-']).rstrip()
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=timeseries_{clean_filename}_{analysis_id[:8]}.csv"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate time-series CSV for analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate time-series CSV data.")

@app.websocket("/ws/live/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    await manager.connect(websocket, analysis_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(analysis_id)
        logger.info(f"WebSocket disconnected for analysis_id: {analysis_id}")

@app.post("/api/analysis/{analysis_id}/correct")
async def correct_analysis_data(analysis_id: str, payload: CorrectionPayload, background_tasks: BackgroundTasks):
    """
    Corrects the analysis data by removing specified time ranges from the raw landmarks
    and re-running the analysis on the corrected data.
    """
    # 1. Fetch existing analysis data
    analysis_record = db_query_one("SELECT results, raw_landmarks_json FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis_record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    original_results = json.loads(analysis_record['results'])
    raw_landmarks_json = analysis_record['raw_landmarks_json']
    
    if not raw_landmarks_json:
        raise HTTPException(status_code=400, detail="Cannot correct analysis without raw landmark data.")

    df = pd.read_json(raw_landmarks_json)

    # 2. Remove the specified time ranges for the specified hand
    corrected_df = df.copy()
    for time_range in payload.time_ranges_to_remove:
        start = time_range.start_time
        end = time_range.end_time
        # Remove data for the specified hand within the time range
        corrected_df = corrected_df[
            ~((corrected_df['hand'] == payload.hand) & (corrected_df['time'] >= start) & (corrected_df['time'] <= end))
        ]

    # 3. Re-run the analysis calculation with the corrected DataFrame
    params = original_results.get('analysis_parameters', {})
    
    # This function now performs the core calculations
    new_results_dict = perform_analysis_on_dataframe(analysis_id, corrected_df, params)

    # The new results from perform_analysis_on_dataframe already contain 'summary' and 'analysis' keys
    original_results['summary'] = new_results_dict.get('summary', original_results.get('summary'))
    original_results['analysis'] = new_results_dict.get('analysis', original_results.get('analysis'))


    # Update the database with the corrected results and new raw landmarks
    db_execute(
        "UPDATE analyses SET results = ?, raw_landmarks_json = ? WHERE id = ?",
        (json.dumps(original_results), corrected_df.to_json(orient='records'), analysis_id)
    )

    return JSONResponse(content=original_results)

@app.get("/api/analysis/{analysis_id}/download_final_values_csv")
async def download_analysis_final_values_csv(analysis_id: str):
    """Download only the final values for all kinematic metrics as CSV"""
    analysis = db_query_one("SELECT results, original_name FROM analyses WHERE id = ?", (analysis_id,))
    if not analysis or not analysis['results']:
        raise HTTPException(status_code=404, detail="Analysis results not found.")
    
    try:
        results = json.loads(analysis['results'])
        summary = results.get('summary', {})
        analysis_data = results.get('analysis', {})
        
        # Prepare CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header with all final value metrics including new ones
        headers = [
            'analysis_id', 'filename', 'analysis_date',
            # Left hand final values
            'left_total_path_length_pixels',
            'left_peak_velocity_pps',
            'left_average_velocity_pps',  # NEW
            'left_peak_acceleration_pps2', 
            'left_average_acceleration_pps2',  # NEW
            'left_peak_jerk_pps3',
            'left_total_cumulative_jerk_pps3s',
            'left_log_dimensionless_jerk',
            'left_dimensionless_jerk',  # NEW
            'left_squared_jerk_integral',  # NEW
            # Right hand final values
            'right_total_path_length_pixels',
            'right_peak_velocity_pps',
            'right_average_velocity_pps',  # NEW
            'right_peak_acceleration_pps2',
            'right_average_acceleration_pps2',  # NEW
            'right_peak_jerk_pps3', 
            'right_total_cumulative_jerk_pps3s',
            'right_log_dimensionless_jerk',
            'right_dimensionless_jerk',  # NEW
            'right_squared_jerk_integral',  # NEW
            # Combined final values
            'combined_total_path_length_pixels',
            'combined_peak_velocity_pps',
            'combined_average_velocity_pps',  # NEW
            'combined_peak_acceleration_pps2',
            'combined_average_acceleration_pps2',  # NEW
            'combined_peak_jerk_pps3',
            'combined_total_cumulative_jerk_pps3s'
        ]
        writer.writerow(headers)
        
        # Extract final values from time series data
        left_summary = summary.get('Left', {})
        right_summary = summary.get('Right', {})
        overall_summary = summary.get('Overall', {})
        
        # Get final cumulative jerk values from time series
        left_final_cumulative_jerk = 0
        right_final_cumulative_jerk = 0
        
        if 'cumulative_jerk' in analysis_data:
            left_cumulative_jerk_data = [d for d in analysis_data['cumulative_jerk'] if d.get('hand') == 'Left']
            right_cumulative_jerk_data = [d for d in analysis_data['cumulative_jerk'] if d.get('hand') == 'Right']
            
            if left_cumulative_jerk_data:
                left_final_cumulative_jerk = max(d.get('Cumulative Jerk', 0) for d in left_cumulative_jerk_data)
            if right_cumulative_jerk_data:
                right_final_cumulative_jerk = max(d.get('Cumulative Jerk', 0) for d in right_cumulative_jerk_data)
        
        # Calculate combined values
        combined_total_path = (left_summary.get('total_path_pixels', 0) + 
                              right_summary.get('total_path_pixels', 0))
        combined_peak_velocity = max(left_summary.get('peak_velocity_pps', 0),
                                   right_summary.get('peak_velocity_pps', 0))
        combined_average_velocity = (left_summary.get('average_velocity_pps', 0) + 
                                   right_summary.get('average_velocity_pps', 0)) / 2
        combined_peak_acceleration = max(left_summary.get('peak_acceleration_pps2', 0),
                                       right_summary.get('peak_acceleration_pps2', 0))
        combined_average_acceleration = (left_summary.get('average_acceleration_pps2', 0) + 
                                       right_summary.get('average_acceleration_pps2', 0)) / 2
        combined_peak_jerk = max(left_summary.get('peak_jerk_pps3', 0),
                               right_summary.get('peak_jerk_pps3', 0))
        combined_cumulative_jerk = left_final_cumulative_jerk + right_final_cumulative_jerk
        
        # Create the data row
        row = [
            analysis_id,
            analysis['original_name'],
            results.get('created_at', ''),
            # Left hand
            left_summary.get('total_path_pixels', ''),
            left_summary.get('peak_velocity_pps', ''),
            left_summary.get('average_velocity_pps', ''),  # NEW
            left_summary.get('peak_acceleration_pps2', ''),
            left_summary.get('average_acceleration_pps2', ''),  # NEW
            left_summary.get('peak_jerk_pps3', ''),
            left_final_cumulative_jerk if left_final_cumulative_jerk > 0 else '',
            left_summary.get('log_dimensionless_jerk', ''),
            left_summary.get('dimensionless_jerk', ''),  # NEW
            left_summary.get('squared_jerk_integral', ''),  # NEW
            # Right hand
            right_summary.get('total_path_pixels', ''),
            right_summary.get('peak_velocity_pps', ''),
            right_summary.get('average_velocity_pps', ''),  # NEW
            right_summary.get('peak_acceleration_pps2', ''),
            right_summary.get('average_acceleration_pps2', ''),  # NEW
            right_summary.get('peak_jerk_pps3', ''),
            right_final_cumulative_jerk if right_final_cumulative_jerk > 0 else '',
            right_summary.get('log_dimensionless_jerk', ''),
            right_summary.get('dimensionless_jerk', ''),  # NEW
            right_summary.get('squared_jerk_integral', ''),  # NEW
            # Combined
            combined_total_path if combined_total_path > 0 else '',
            combined_peak_velocity if combined_peak_velocity > 0 else '',
            combined_average_velocity if combined_average_velocity > 0 else '',  # NEW
            combined_peak_acceleration if combined_peak_acceleration > 0 else '',
            combined_average_acceleration if combined_average_acceleration > 0 else '',  # NEW
            combined_peak_jerk if combined_peak_jerk > 0 else '',
            combined_cumulative_jerk if combined_cumulative_jerk > 0 else ''
        ]
        
        writer.writerow(row)
        
        output.seek(0)
        clean_filename = "".join([c for c in analysis['original_name'] if c.isalpha() or c.isdigit() or c in '._-']).rstrip()
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=final_values_{clean_filename}_{analysis_id[:8]}.csv"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate final values CSV for analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate final values CSV data.")

# --- Statistics Endpoints (removed by request) ---
"""
The following statistics endpoints and helpers were removed per user request.
"""
def _parse_date(date_str: str) -> str:
    """Validate simple YYYY-MM-DD date and convert to SQLite comparable ranges."""
    try:
        # Accept only simple date format
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

def _load_completed_analyses(from_date: Optional[str], to_date: Optional[str]):
    query = "SELECT id, created_at, results FROM analyses WHERE status = 'completed'"
    params: list[str] = []
    if from_date:
        query += " AND date(created_at) >= date(?)"
        params.append(from_date)
    if to_date:
        query += " AND date(created_at) <= date(?)"
        params.append(to_date)
    query += " ORDER BY created_at ASC"
    rows = db_query_all(query, tuple(params))
    analyses = []
    for row in rows:
        try:
            results = json.loads(row["results"]) if row["results"] else {}
        except Exception:
            results = {}
        analyses.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "results": results,
        })
    return analyses

def _collect_hand_metric_values(analyses: List[dict], hand: str, metric: str) -> List[float]:
    values: List[float] = []
    for a in analyses:
        summary = a.get("results", {}).get("summary", {})
        hand_summary = summary.get(hand, {})
        v = hand_summary.get(metric)
        if isinstance(v, (int, float)):
            values.append(float(v))
    return values

def _collect_overall_dexterity(analyses: List[dict]) -> List[float]:
    values: List[float] = []
    for a in analyses:
        overall = a.get("results", {}).get("summary", {}).get("Overall", {})
        v = overall.get("dexterity_score")
        if isinstance(v, (int, float)):
            values.append(float(v))
    return values

def _summary_stats(values: List[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "p25": None, "p75": None, "min": None, "max": None}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

@app.get("/api/stats/summary")
async def stats_summary(from_date: Optional[str] = None, to_date: Optional[str] = None, hand: Optional[str] = None):
    if from_date:
        from_date = _parse_date(from_date)
    if to_date:
        to_date = _parse_date(to_date)

    analyses = _load_completed_analyses(from_date, to_date)
    metrics = [
        "total_path_pixels",
        "peak_velocity_pps",
        "average_velocity_pps",
        "peak_acceleration_pps2",
        "average_acceleration_pps2",
        "peak_jerk_pps3",
        "total_cumulative_jerk",
        "log_dimensionless_jerk",
        "dimensionless_jerk",
        "squared_jerk_integral",
    ]

    hands = [hand] if hand in ("Left", "Right") else ["Left", "Right"]
    hand_summaries: dict[str, dict] = {}
    for h in hands:
        hand_summaries[h] = {m: _summary_stats(_collect_hand_metric_values(analyses, h, m)) for m in metrics}

    overall_values = _collect_overall_dexterity(analyses)

    return {
        "filters": {"from": from_date, "to": to_date, "hand": hand},
        "counts": {"total_completed": len(analyses)},
        "overall_dexterity": _summary_stats(overall_values),
        "hand_summaries": hand_summaries,
    }

@app.get("/api/stats/distribution")
async def stats_distribution(metric: str, hand: str = "Left", bins: int = 20, from_date: Optional[str] = None, to_date: Optional[str] = None):
    if hand not in ("Left", "Right"):
        raise HTTPException(status_code=400, detail="hand must be 'Left' or 'Right'")
    if from_date:
        from_date = _parse_date(from_date)
    if to_date:
        to_date = _parse_date(to_date)

    analyses = _load_completed_analyses(from_date, to_date)
    values = _collect_hand_metric_values(analyses, hand, metric)
    if not values:
        return {"metric": metric, "hand": hand, "bins": [], "counts": []}
    counts, bin_edges = np.histogram(values, bins=bins)
    return {
        "metric": metric,
        "hand": hand,
        "bins": [float(b) for b in bin_edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "count": len(values)
    }

@app.get("/api/stats/trends")
async def stats_trends(metric: str = "dexterity_score_overall", interval: str = "daily", from_date: Optional[str] = None, to_date: Optional[str] = None, hand: Optional[str] = None):
    if from_date:
        from_date = _parse_date(from_date)
    if to_date:
        to_date = _parse_date(to_date)
    if interval not in ("daily", "weekly", "monthly"):
        raise HTTPException(status_code=400, detail="interval must be daily|weekly|monthly")

    analyses = _load_completed_analyses(from_date, to_date)

    # Extract values per analysis date key
    buckets: dict[str, list[float]] = {}
    for a in analyses:
        # Determine bucket key
        try:
            dt = datetime.fromisoformat(a["created_at"])  # created_at is SQLite timestamp
        except Exception:
            try:
                dt = datetime.strptime(a["created_at"], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
        if interval == "daily":
            key = dt.strftime("%Y-%m-%d")
        elif interval == "weekly":
            key = f"{dt.strftime('%Y')}-W{dt.isocalendar().week:02d}"
        else:
            key = dt.strftime("%Y-%m")

        # Pick value
        if metric == "dexterity_score_overall":
            v_list = _collect_overall_dexterity([a])
        else:
            # Expect format: metricName:hand (e.g., peak_velocity_pps:Left)
            if ":" in metric:
                mname, h = metric.split(":", 1)
            else:
                mname, h = metric, (hand or "Left")
            v_list = _collect_hand_metric_values([a], h, mname)
        if v_list:
            buckets.setdefault(key, []).extend(v_list)

    # Aggregate mean per bucket
    series = [{"bucket": k, "mean": float(np.mean(vs)), "count": len(vs)} for k, vs in sorted(buckets.items())]
    return {"metric": metric, "interval": interval, "series": series}

@app.get("/api/stats/summary_csv")
async def stats_summary_csv(from_date: Optional[str] = None, to_date: Optional[str] = None):
    if from_date:
        from_date = _parse_date(from_date)
    if to_date:
        to_date = _parse_date(to_date)
    analyses = _load_completed_analyses(from_date, to_date)
    metrics = [
        "total_path_pixels",
        "peak_velocity_pps",
        "average_velocity_pps",
        "peak_acceleration_pps2",
        "average_acceleration_pps2",
        "peak_jerk_pps3",
        "total_cumulative_jerk",
        "log_dimensionless_jerk",
        "dimensionless_jerk",
        "squared_jerk_integral",
    ]
    output = _io.StringIO()
    writer = csv.writer(output)
    header = ["hand", "metric", "count", "mean", "median", "std", "p25", "p75", "min", "max"]
    writer.writerow(header)
    for hand in ["Left", "Right"]:
        for m in metrics:
            stats_obj = _summary_stats(_collect_hand_metric_values(analyses, hand, m))
            writer.writerow([
                hand, m,
                stats_obj["count"],
                stats_obj["mean"],
                stats_obj["median"],
                stats_obj["std"],
                stats_obj["p25"],
                stats_obj["p75"],
                stats_obj["min"],
                stats_obj["max"],
            ])
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=stats_summary.csv"})

@app.get("/api/stats/correlation")
async def stats_correlation(hand: Optional[str] = "Left", method: str = "spearman", from_date: Optional[str] = None, to_date: Optional[str] = None):
    if hand not in ("Left", "Right"):
        raise HTTPException(status_code=400, detail="hand must be 'Left' or 'Right'")
    if method not in ("pearson", "spearman"):
        raise HTTPException(status_code=400, detail="method must be pearson|spearman")
    if from_date:
        from_date = _parse_date(from_date)
    if to_date:
        to_date = _parse_date(to_date)
    analyses = _load_completed_analyses(from_date, to_date)
    metrics = [
        "total_path_pixels",
        "peak_velocity_pps",
        "average_velocity_pps",
        "peak_acceleration_pps2",
        "average_acceleration_pps2",
        "peak_jerk_pps3",
        "total_cumulative_jerk",
        "log_dimensionless_jerk",
        "dimensionless_jerk",
        "squared_jerk_integral",
    ]
    # Build matrix
    rows = []
    for a in analyses:
        row = []
        hs = a.get("results", {}).get("summary", {}).get(hand, {})
        for m in metrics:
            v = hs.get(m)
            row.append(float(v) if isinstance(v, (int, float)) else np.nan)
        rows.append(row)
    if not rows:
        return {"hand": hand, "metrics": metrics, "correlation": []}
    data = np.array(rows, dtype=float)
    # Remove rows with all nan
    data = data[~np.isnan(data).all(axis=1)]
    if data.shape[0] < 2:
        return {"hand": hand, "metrics": metrics, "correlation": []}
    if method == "pearson":
        corr = np.corrcoef(np.nan_to_num(data, nan=np.nanmean(data, axis=0)), rowvar=False)
    else:
        # Spearman: rank transform per column, then Pearson on ranks
        ranks = np.apply_along_axis(lambda col: _stats.rankdata(col, nan_policy='omit'), 0, data)
        corr = np.corrcoef(np.nan_to_num(ranks, nan=np.nanmean(ranks, axis=0)), rowvar=False)
    # Clip numerical noise to [-1,1]
    corr = np.clip(corr, -1.0, 1.0)
    return {"hand": hand, "method": method, "metrics": metrics, "correlation": corr.tolist()}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(MODEL_PATH):
        logger.error("-" * 50)
        logger.error("CRITICAL ERROR: The hand landmark model is missing.")
        logger.error(f"Please download 'hand_landmarker.task' from:\nhttps://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        logger.error(f"And place it in: {os.path.abspath(MODELS_DIR)}")
        logger.error("-" * 50)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 