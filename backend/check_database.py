#!/usr/bin/env python3
"""Check database state for race history feature"""
import os
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')

if not url or not key:
    print('ERROR: SUPABASE credentials missing')
    exit(1)

try:
    client = create_client(url, key)
    
    # Races by year
    print("\n" + "="*50)
    print("RACES TABLE - BY YEAR")
    print("="*50)
    races_result = client.table('races').select('race_year').execute()
    years = {}
    for row in races_result.data:
        y = row['race_year']
        years[y] = years.get(y, 0) + 1
    
    if years:
        for y in sorted(years.keys(), reverse=True)[:10]:
            print(f"  {y}: {years[y]} entries")
    else:
        print("  [EMPTY - NO RACES IN DATABASE]")
    
    # Qualifying cache
    print("\n" + "="*50)
    print("QUALIFYING_CACHE TABLE")
    print("="*50)
    qual = client.table('qualifying_cache').select('race_key, cached_at').execute()
    print(f"Total entries: {len(qual.data)}")
    if qual.data:
        for row in qual.data[:5]:
            print(f"  - {row['race_key']} (cached: {row['cached_at'][:10]})")
    else:
        print("  [EMPTY]")
    
    # Drivers
    print("\n" + "="*50)
    print("DRIVERS TABLE")
    print("="*50)
    drivers = client.table('drivers').select('code, current_team').execute()
    print(f"Total drivers: {len(drivers.data)}")
    
    # Group by team
    teams = {}
    for d in drivers.data:
        team = d.get('current_team', 'Unknown')
        teams[team] = teams.get(team, 0) + 1
    
    print(f"Active teams: {len(teams)}")
    for team in sorted(teams.keys())[:5]:
        print(f"  - {team}: {teams[team]} drivers")
    
    # Predictions
    print("\n" + "="*50)
    print("PREDICTIONS TABLE")
    print("="*50)
    preds = client.table('predictions').select('id').execute()
    print(f"Total predictions: {len(preds.data)}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Races in database: {sum(years.values())} total")
    print(f"2025 races: {years.get(2025, 0)}")
    print(f"Qualifying cache entries: {len(qual.data)}")
    print(f"Drivers in database: {len(drivers.data)}")
    
except Exception as e:
    print(f'\nError: {e}')
    import traceback
    traceback.print_exc()
