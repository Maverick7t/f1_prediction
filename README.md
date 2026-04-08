# F1 Race Prediction System

Full-stack F1 race prediction app with a Flask API + XGBoost models and a React (Vite) frontend.

## Quick start (Windows)

```powershell
# From repo root
.\start.ps1
```

Backend: http://localhost:5000
Frontend: http://localhost:5173

## Manual run

### Backend

```powershell
cd backend
python -m venv f1env
.\f1env\Scripts\Activate.ps1
pip install -r requirements.txt
python app/run_dev.py
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Config (minimal)

- Frontend: `VITE_API_URL` (defaults to `http://localhost:5000`)
- Backend: see `backend/config/development.env` for typical local defaults

## API

- `GET /api/health` — health check
- `GET /api/next-race` — next race info
- `GET /api/qualifying` — qualifying data
- `POST /api/predict` — run a prediction
- `GET /api/prediction-history` — recent predictions
- `POST /api/update-actual-winner` — backfill actual winner + correctness

## Deployment

- Render: see `render.yaml` and `RENDER_DEPLOYMENT.md`
- Vercel: deploy `frontend/` and set `VITE_API_URL` to your backend URL

## License

MIT
