"""
Store 2025 Season Predictions to Supabase

This script:
1. Loads the ML models and historical data
2. Runs get_season_review(2025) to generate predictions for each race
3. Stores each race prediction into `predictions` table in Supabase
4. Also stores the full season review summary as a JSON row

After running this once, the predictions are persisted in the database
and the frontend can read them instantly without recomputing.
"""

import os
import sys
import json
from datetime import datetime

# Setup paths
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(backend_dir, '.env'))

from supabase import create_client

def main():
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not url or not key:
        print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env")
        sys.exit(1)
    
    sb = create_client(url, key)
    print(f"✓ Connected to Supabase: {url}")
    
    # Check current state
    existing = sb.table('predictions').select('id', count='exact').execute()
    print(f"  Current predictions in DB: {existing.count}")
    
    # Import the API module which loads models and data
    print("Loading ML models and historical data...")
    from app.api import get_season_review
    print("✓ Models loaded")
    
    # Generate 2025 season review
    print("\n=== Generating 2025 Season Review ===")
    review = get_season_review(2025)
    
    if not review or not review.get('races'):
        print("❌ No season review data generated")
        sys.exit(1)
    
    races = review['races']
    stats = review.get('stats', {})
    
    print(f"✓ Generated predictions for {len(races)} races")
    print(f"  Winner accuracy: {stats.get('accuracy_percentage', 0)}%")
    print(f"  Podium accuracy: {stats.get('podium_accuracy_percentage', 0)}%")
    
    # Clear existing 2025 predictions
    print("\nClearing existing 2025 predictions...")
    try:
        sb.table('predictions').delete().eq('race_year', 2025).execute()
        print("✓ Cleared old 2025 predictions")
    except Exception as e:
        print(f"  Warning: Could not clear old predictions: {e}")
    
    # Store each race prediction
    print("\n=== Storing predictions in Supabase ===")
    stored = 0
    for race in races:
        try:
            row = {
                'race': race['race'],
                'race_year': 2025,
                'circuit': race.get('circuit', ''),
                'predicted': race.get('predicted_winner', 'N/A'),
                'confidence': race.get('confidence', 0),
                'model_version': 'v3',
                'actual': race.get('actual_winner', ''),
                'correct': race.get('correct', False),
                'full_predictions': json.dumps({
                    'round': race.get('round', 0),
                    'date': race.get('date', ''),
                    'predicted_top3': race.get('predicted_top3', []),
                    'actual_podium': race.get('actual_podium', []),
                    'podium_correct': race.get('podium_correct', False),
                    'status': race.get('status', 'unknown')
                }),
                'timestamp': race.get('date', datetime.now().isoformat())
            }
            
            sb.table('predictions').insert(row).execute()
            stored += 1
            
            status = "✓" if race.get('correct') else "✗"
            print(f"  {status} R{race.get('round', '?'):>2}: {race['race']:<35} Predicted={race.get('predicted_winner', 'N/A'):<5} Actual={race.get('actual_winner', '?'):<5}")
            
        except Exception as e:
            print(f"  ❌ Failed to store {race['race']}: {e}")
    
    # Store the season summary as a special row
    try:
        summary_row = {
            'race': f'__SEASON_SUMMARY_2025__',
            'race_year': 2025,
            'circuit': 'ALL',
            'predicted': f"{stats.get('correct_predictions', 0)}/{stats.get('total_races', 0)}",
            'confidence': stats.get('accuracy_percentage', 0),
            'model_version': 'v3',
            'actual': f"podium:{stats.get('podium_correct', 0)}/{stats.get('total_races', 0)}",
            'correct': stats.get('accuracy_percentage', 0) > 50,
            'full_predictions': json.dumps({
                'stats': stats,
                'available_years': review.get('available_years', []),
                'generated_at': datetime.now().isoformat()
            }),
            'timestamp': datetime.now().isoformat()
        }
        sb.table('predictions').insert(summary_row).execute()
        stored += 1
        print(f"\n✓ Stored season summary row")
    except Exception as e:
        print(f"\n  ❌ Failed to store season summary: {e}")
    
    # Verify
    final = sb.table('predictions').select('id', count='exact').eq('race_year', 2025).execute()
    print(f"\n{'='*50}")
    print(f"✓ DONE: {stored} rows stored in predictions table")
    print(f"  Total 2025 predictions in DB: {final.count}")
    print(f"  Winner accuracy: {stats.get('accuracy_percentage', 0)}% ({stats.get('correct_predictions', 0)}/{stats.get('total_races', 0)})")
    print(f"  Podium accuracy: {stats.get('podium_accuracy_percentage', 0)}% ({stats.get('podium_correct', 0)}/{stats.get('total_races', 0)})")


if __name__ == '__main__':
    main()
