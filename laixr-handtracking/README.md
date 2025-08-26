# 🖐️ LAIXR Surgical Hand Tracking Analysis Platform

A professional web-based application for detailed analysis of surgical hand movements from video recordings. Built with MediaPipe for precise hand landmark detection and provides comprehensive metrics for evaluating surgical performance.

![Platform Preview](docs/screenshot.png) <!-- Add a screenshot -->

## ✨ Features

- **Real-time Hand Tracking**: Advanced MediaPipe integration for precise landmark detection
- **Comprehensive Metrics**: Dexterity scores, speed analysis, acceleration, jerk calculations
- **Professional Analytics**: Dimensionless jerk, path analysis, and movement smoothness
- **Export Capabilities**: Detailed CSV exports with time-series data
- **Live Analysis**: Real-time video processing with live feedback
- **Cross-platform**: Runs on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v18 or later) - [Download here](https://nodejs.org/)
- **Python** (v3.9 or later) - [Download here](https://www.python.org/)

### 📦 One-Command Setup

**On Windows:**
```bash
setup.bat
```

**On macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### ▶️ Running the Application

**On Windows:**
```bash
start.bat
```

**On macOS/Linux:**
```bash
./start.sh
```

The application will be available at **http://localhost:3000**

Set environment variable for frontend-backend communication:

Windows PowerShell:
```powershell
$env:NEXT_PUBLIC_BACKEND_URL="http://localhost:8000"
```
macOS/Linux:
```bash
export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### ⏹️ Stopping the Application

Press **`CTRL+C`** in the terminal to stop both servers.

## 📁 Project Structure

- **/backend**: Contains the FastAPI server, video analysis logic (`main.py`), and Python dependencies (`requirements.txt`