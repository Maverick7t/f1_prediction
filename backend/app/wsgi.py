"""
WSGI Entry Point for Production
Use with Gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
"""

from api import app

if __name__ == "__main__":
    app.run()
