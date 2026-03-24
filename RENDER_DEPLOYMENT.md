# Render Deployment Configuration

## Critical: Set Python Version in Render Environment

Render ignores `runtime.txt` and defaults to Python 3.13.4. This causes compatibility issues with older dependencies.

### Action Required:
1. Go to your Render Service Dashboard
2. Navigate to **Settings** → **Environment**
3. Add/Update the environment variable:
   ```
   PYTHON_VERSION = 3.12.5
   ```
4. Click **Save Changes**
5. Trigger a **Manual Deploy** → **Deploy latest commit**

### Why This Matters:
- Python 3.12.5 has prebuilt wheels for `pyarrow 14.x`
- `mlflow 2.9.1` requires `pyarrow <15`
- Python 3.13.4 lacks wheels, forcing source compilation and `pkg_resources` errors

### Verification:
After deployment, check logs for:
```
==> Installing Python version 3.12.5...
==> Using Python version 3.12.5 (default)
```

## Alternative: Upgrade MLflow (Future)
To support Python 3.13.4 without environment variables:
```
mlflow>=3.11.0  # Supports pyarrow>=15 with Python 3.13 wheels
```

## Current Status
- ✅ Fixed: `/api/next-race` uses dynamic year (2026, not hardcoded 2025)
- ⏳ Pending: PYTHON_VERSION=3.12.5 environment variable in Render
- ✅ Updated: `pyarrow>=14.0.1,<15` to match mlflow 2.9.1 constraint
