#!/usr/bin/env python
"""Quick test of the prediction endpoint"""
import requests
import json

try:
    response = requests.get('http://localhost:5000/api/predict/sao-paulo', timeout=30)
    print(f'Status: {response.status_code}')
    data = response.json()
    print(f'\nResponse keys: {list(data.keys())}')
    
    if 'next_race' in data and data['next_race']:
        nr = data['next_race']
        print(f'\nNext race keys: {list(nr.keys())}')
        print(f'Has full_predictions: {"full_predictions" in nr}')
        if 'full_predictions' in nr:
            print(f'Full predictions count: {len(nr["full_predictions"])}')
            if nr['full_predictions']:
                print(f'First prediction: {nr["full_predictions"][0]}')
    else:
        print('No next_race in response')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
