"""
Setup Supabase tables for the new project.
Run this once to create all required tables.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(override=True)

from supabase import create_client

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY')

print(f"Connecting to: {url}")
sb = create_client(url, key)

# Test connection
print("Testing connection...")
try:
    # Simple RPC call to test
    result = sb.rpc('version', {}).execute()
    print(f"Connected! Postgres version available.")
except Exception:
    # Try a simple table query instead
    pass

# ===== CREATE TABLES VIA SQL =====
# Supabase Python client doesn't support raw SQL directly.
# Tables need to be created via the Supabase SQL Editor.
# 
# Instead, let's verify which tables exist and create data.

print("\n=== Checking tables ===")

tables_to_check = ['predictions', 'qualifying_cache', 'standings_cache', 'races', 'drivers', 'teams', 'circuits', 'model_registry']
existing = []
missing = []

for table in tables_to_check:
    try:
        r = sb.table(table).select('*').limit(1).execute()
        existing.append(table)
        print(f"  ✓ {table} exists ({len(r.data)} rows)")
    except Exception as e:
        err = str(e)
        if '42P01' in err or 'does not exist' in err.lower() or 'relation' in err.lower():
            missing.append(table)
            print(f"  ✗ {table} - NOT FOUND")
        else:
            print(f"  ? {table} - {err[:100]}")
            missing.append(table)

print(f"\nExisting: {existing}")
print(f"Missing: {missing}")

if missing:
    print(f"\n{'='*60}")
    print("ACTION REQUIRED: Create missing tables in Supabase SQL Editor")
    print(f"{'='*60}")
    print(f"\n1. Go to https://supabase.com/dashboard/project/xoujvbujweonptqjyzcc/sql")
    print(f"2. Copy and paste the SQL from: backend/database/schema.sql")
    print(f"3. Click 'Run'")
    print(f"4. Then re-run this script to verify")
else:
    print("\n✓ All tables exist!")

# If predictions table exists, show count
if 'predictions' in existing:
    try:
        r = sb.table('predictions').select('*', count='exact').execute()
        print(f"\nPredictions in DB: {r.count if hasattr(r, 'count') else len(r.data)}")
    except:
        pass

if 'qualifying_cache' in existing:
    try:
        r = sb.table('qualifying_cache').select('race_key, race_year').execute()
        print(f"Qualifying cache entries: {len(r.data)}")
        for row in r.data[:5]:
            print(f"  {row}")
    except:
        pass

if 'races' in existing:
    try:
        r = sb.table('races').select('race_year').execute()
        if r.data:
            years = set(row['race_year'] for row in r.data)
            print(f"Race data years: {sorted(years)}")
            print(f"Total race entries: {len(r.data)}")
    except:
        pass

print("\nDone!")
