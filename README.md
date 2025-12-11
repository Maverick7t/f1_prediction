# F1 Race Prediction System 🏎️

Full-stack F1 race prediction application using XGBoost ML models and React frontend.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/f1_hackathon)

## 🏗️ Architecture

Professional, scalable 3-tier architecture:

```
f1_hackathon/
├── backend/                    # Python Flask API (see BACKEND_STRUCTURE.md)
│   ├── app/                   # Main application (Flask server)
│   ├── database/              # Supabase database layer
│   ├── services/              # Business logic & ML services
│   ├── utils/                 # Configuration & helpers
│   ├── scripts/               # Administrative scripts
│   ├── tests/                 # Unit tests & notebooks
│   ├── models_spencer/        # Trained ML models
│   ├── data/                  # Training data
│   ├── .env                   # Environment variables (not in git)
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # React + Vite + Tailwind
│   ├── src/
│   │   ├── api.js             # API client
│   │   └── components/        # React components
│   ├── .env                   # Frontend config
│   └── vercel.json            # Vercel deployment config
│
├── testing/                    # Integration & E2E tests
├── docs/                       # Documentation
├── render.yaml                 # Render deployment config
├── BACKEND_STRUCTURE.md        # Detailed backend organization
└── start.ps1                   # Local development launcher
```

## 🚀 Quick Start

### Option 1: One-Command Start (Windows)

```powershell
# From project root
.\start.ps1
```

This will:
- Set up Python virtual environment
- Install all dependencies
- Start backend on http://localhost:5000
- Start frontend on http://localhost:5173
- Open browser automatically

### Option 2: Manual Setup

#### Backend Setup

```powershell
cd backend

# Create virtual environment
python -m venv f1env

# Activate (Windows)
.\f1env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start server
python api.py
```

#### Frontend Setup

```powershell
cd frontend

# Install dependencies
npm install

# Copy environment config (optional)
cp .env.example .env

# Start dev server
npm run dev
```

## ⚙️ Configuration

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment mode | `development` |
| `SECRET_KEY` | Flask secret key | Auto-generated |
| `DATA_PATH` | Training data CSV | `./f1_training_dataset_2018_2024.csv` |
| `MODEL_DIR` | Model files directory | `./models_spencer` |
| `CACHE_DIR` | Cache directory | `./cache` |
| `FASTF1_CACHE_DIR` | FastF1 cache | `./f1_cache` |
| `DATABASE_URL` | PostgreSQL URL (production) | - |
| `REDIS_URL` | Redis URL (production) | - |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:5000` |

## 🌐 Deployment

### Deploy Backend to Render (Free)

1. Push code to GitHub
2. Click "Deploy to Render" button above, OR:
3. Go to [render.com](https://render.com)
4. New → Web Service → Connect your repo
5. Render auto-detects `render.yaml` configuration

### Deploy Frontend to Vercel (Free)

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your repository
4. Set root directory to `frontend`
5. Add environment variable:
   - `VITE_API_URL` = Your Render backend URL

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/predict` | POST | Custom race prediction |
| `/api/predict/sao-paulo` | GET | São Paulo GP prediction |
| `/api/qualifying` | GET | Get qualifying data |
| `/api/next-race` | GET | Next race info |
| `/api/driver-standings` | GET | Driver standings |
| `/api/constructor-standings` | GET | Constructor standings |
| `/api/race-history` | GET | Recent race history |
| `/api/model-registry` | GET | MLflow model info |
| `/api/model-metrics` | GET | Prediction accuracy |

## 🤖 ML Models

- **XGBoost Winner Model**: Predicts race winner probability
- **XGBoost Podium Model**: Predicts top 3 finish probability
- **Training Data**: 2018-2024 F1 race results
- **Features**: Qualifying position, ELO rating, recent form, team performance

## 🛠️ Tech Stack

**Backend:**
- Python 3.11
- Flask + Gunicorn
- XGBoost + scikit-learn
- MLflow (experiment tracking)
- FastF1 (telemetry data)

**Frontend:**
- React 19
- Vite 7
- Tailwind CSS 4
- Leaflet (maps)

**External APIs:**
- FastF1 API
- Ergast F1 API
- OpenF1 API

## 📄 License

MIT License
