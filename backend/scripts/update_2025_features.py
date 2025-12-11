"""
F1 2025 Feature Update Script
Fetches current 2025 season data and updates feature values for predictions.

This script:
1. Fetches 2025 race results from Ergast API
2. Calculates current Elo ratings, recent form, team performance
3. Saves to Supabase AND updates local parquet snapshots
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONSTANTS
# =============================================================================
ERGAST_BASE = "https://ergast.com/api/f1"
DEFAULT_YEAR = 2025

# Elo constants
ELO_K_FACTOR = 32
ELO_BASE = 1500

# Circuit name mappings (Ergast full names → Our simplified names)
CIRCUIT_MAPPINGS = {
    "Bahrain International Circuit": "Bahrain",
    "Jeddah Corniche Circuit": "Saudi Arabia",
    "Albert Park Grand Prix Circuit": "Australia",
    "Suzuka Circuit": "Japan",
    "Shanghai International Circuit": "China",
    "Miami International Autodrome": "Miami",
    "Autodromo Enzo e Dino Ferrari": "Imola",
    "Circuit de Monaco": "Monaco",
    "Circuit Gilles Villeneuve": "Canada",
    "Circuit de Barcelona-Catalunya": "Spain",
    "Red Bull Ring": "Austria",
    "Silverstone Circuit": "Silverstone",
    "Hungaroring": "Hungary",
    "Circuit de Spa-Francorchamps": "Belgium",
    "Circuit Park Zandvoort": "Netherlands",
    "Autodromo Nazionale di Monza": "Monza",
    "Baku City Circuit": "Azerbaijan",
    "Marina Bay Street Circuit": "Singapore",
    "Circuit of the Americas": "USA",
    "Autodromo Hermanos Rodriguez": "Mexico",
    "Autodromo Jose Carlos Pace": "Brazil",
    "Interlagos": "Brazil",
    "Las Vegas Strip Street Circuit": "Las Vegas",
    "Lusail International Circuit": "Qatar",
    "Yas Marina Circuit": "Abu Dhabi",
}

# Team name mappings (Ergast → Our naming)
TEAM_MAPPINGS = {
    "Red Bull": "Red Bull",
    "Red Bull Racing": "Red Bull",
    "red_bull": "Red Bull",
    "McLaren": "McLaren",
    "mclaren": "McLaren",
    "Ferrari": "Ferrari",
    "ferrari": "Ferrari",
    "Mercedes": "Mercedes",
    "mercedes": "Mercedes",
    "Aston Martin": "Aston Martin",
    "aston_martin": "Aston Martin",
    "Alpine F1 Team": "Alpine",
    "alpine": "Alpine",
    "Williams": "Williams",
    "williams": "Williams",
    "RB F1 Team": "Racing Bulls",
    "RB": "Racing Bulls",
    "rb": "Racing Bulls",
    "Kick Sauber": "Kick Sauber",
    "Sauber": "Kick Sauber",
    "sauber": "Kick Sauber",
    "Haas F1 Team": "Haas",
    "haas": "Haas",
}

# Driver encodings (matching training data) - mutable to add new drivers
DRIVER_ENCODINGS = {
    "VER": 0, "HAM": 1, "LEC": 2, "NOR": 3, "SAI": 4,
    "RUS": 5, "PER": 6, "ALO": 7, "STR": 8, "GAS": 9,
    "OCO": 10, "TSU": 11, "RIC": 12, "ALB": 13, "MAG": 14,
    "HUL": 15, "BOT": 16, "ZHO": 17, "SAR": 18, "PIA": 19,
    "LAW": 20, "BEA": 21, "COL": 22, "HAD": 23, "ANT": 24,
    "DOO": 25, "BOR": 26
}

TEAM_ENCODINGS = {
    "Red Bull": 0, "McLaren": 1, "Ferrari": 2, "Mercedes": 3,
    "Aston Martin": 4, "Alpine": 5, "Williams": 6, "Racing Bulls": 7,
    "Kick Sauber": 8, "Haas": 9
}

CIRCUIT_ENCODINGS = {
    "Bahrain": 0, "Saudi Arabia": 1, "Australia": 2, "Japan": 3,
    "China": 4, "Miami": 5, "Imola": 6, "Monaco": 7,
    "Canada": 8, "Spain": 9, "Austria": 10, "Silverstone": 11,
    "Hungary": 12, "Belgium": 13, "Netherlands": 14, "Monza": 15,
    "Azerbaijan": 16, "Singapore": 17, "USA": 18, "Mexico": 19,
    "Brazil": 20, "Las Vegas": 21, "Qatar": 22, "Abu Dhabi": 23
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_driver_encoding(driver_code: str) -> int:
    """Get or create driver encoding, handling unknown drivers"""
    if driver_code not in DRIVER_ENCODINGS:
        new_enc = len(DRIVER_ENCODINGS)
        DRIVER_ENCODINGS[driver_code] = new_enc
        print(f"  ⚠️ New driver found: {driver_code} → encoding {new_enc}")
    return DRIVER_ENCODINGS[driver_code]


def get_team_encoding(team_name: str) -> int:
    """Get or create team encoding, handling unknown teams"""
    normalized_team = TEAM_MAPPINGS.get(team_name, team_name)
    if normalized_team not in TEAM_ENCODINGS:
        new_enc = len(TEAM_ENCODINGS)
        TEAM_ENCODINGS[normalized_team] = new_enc
        print(f"  ⚠️ New team found: {normalized_team} → encoding {new_enc}")
    return TEAM_ENCODINGS[normalized_team]


def normalize_circuit_name(circuit_raw: str) -> str:
    """Convert Ergast circuit name to our simplified name"""
    # First try exact match
    if circuit_raw in CIRCUIT_MAPPINGS:
        return CIRCUIT_MAPPINGS[circuit_raw]
    
    # Try fuzzy matching
    circuit_lower = circuit_raw.lower()
    for ergast_name, our_name in CIRCUIT_MAPPINGS.items():
        if ergast_name.lower() in circuit_lower or circuit_lower in ergast_name.lower():
            return our_name
    
    # Fallback: simplify the name
    simplified = circuit_raw.replace(" Circuit", "").replace(" Grand Prix", "")
    print(f"  ⚠️ Unknown circuit: {circuit_raw} → using '{simplified}'")
    return simplified


def get_circuit_encoding(circuit_name: str) -> int:
    """Get circuit encoding, with fallback for unknown circuits"""
    if circuit_name in CIRCUIT_ENCODINGS:
        return CIRCUIT_ENCODINGS[circuit_name]
    
    # Try fuzzy match
    circuit_lower = circuit_name.lower()
    for name, enc in CIRCUIT_ENCODINGS.items():
        if name.lower() in circuit_lower or circuit_lower in name.lower():
            return enc
    
    # Unknown circuit - use a default high encoding
    return len(CIRCUIT_ENCODINGS)


# =============================================================================
# API FUNCTIONS
# =============================================================================
def fetch_ergast(endpoint: str) -> dict:
    """Fetch data from Ergast API"""
    url = f"{ERGAST_BASE}/{endpoint}.json"
    print(f"  Fetching: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_race_results(year: int) -> list:
    """Fetch all race results for given year"""
    print(f"\n📊 Fetching {year} race results...")
    
    try:
        data = fetch_ergast(f"{year}/results")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        print(f"  ✓ Found {len(races)} completed races in {year}")
        return races
    except Exception as e:
        print(f"  ❌ Error fetching results: {e}")
        return []


def get_driver_standings(year: int) -> list:
    """Fetch driver standings for given year"""
    print(f"\n📊 Fetching {year} driver standings...")
    
    try:
        data = fetch_ergast(f"{year}/driverStandings")
        standings = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if standings:
            drivers = standings[0].get("DriverStandings", [])
            print(f"  ✓ Found standings for {len(drivers)} drivers")
            return drivers
        return []
    except Exception as e:
        print(f"  ❌ Error fetching standings: {e}")
        return []


def get_constructor_standings(year: int) -> list:
    """Fetch constructor standings for given year"""
    print(f"\n📊 Fetching {year} constructor standings...")
    
    try:
        data = fetch_ergast(f"{year}/constructorStandings")
        standings = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
        if standings:
            constructors = standings[0].get("ConstructorStandings", [])
            print(f"  ✓ Found standings for {len(constructors)} constructors")
            return constructors
        return []
    except Exception as e:
        print(f"  ❌ Error fetching constructor standings: {e}")
        return []


# =============================================================================
# FEATURE CALCULATION FUNCTIONS
# =============================================================================
def calculate_elo_ratings(race_results: list, base_elo: dict = None) -> dict:
    """
    Calculate Elo ratings based on race results
    
    Uses proper Elo calculation without dividing by number of positions.
    Compares only adjacent positions to avoid inflating updates.
    
    Args:
        race_results: List of race results from Ergast
        base_elo: Starting Elo ratings (from 2024 if available)
    
    Returns:
        Dictionary of driver_code -> elo_rating
    """
    print("\n🎯 Calculating Elo ratings...")
    
    # Initialize with base ratings or default
    elo = base_elo.copy() if base_elo else {}
    
    for race in race_results:
        race_name = race.get("raceName", "Unknown")
        results = race.get("Results", [])
        
        # Get finishing positions sorted by position
        positions = []
        for result in results:
            driver_code = result.get("Driver", {}).get("code", "UNK")
            position = int(result.get("position", 99))
            
            # Initialize driver if not exists
            if driver_code not in elo:
                elo[driver_code] = ELO_BASE
            
            positions.append((driver_code, position))
        
        # Sort by finishing position
        positions.sort(key=lambda x: x[1])
        
        # Update Elo based on ADJACENT position comparisons only
        # This prevents the K-factor from becoming too small
        for i in range(len(positions) - 1):
            d1, p1 = positions[i]
            d2, p2 = positions[i + 1]
            
            # Expected score for d1 (the one who finished ahead)
            expected_d1 = 1 / (1 + 10 ** ((elo[d2] - elo[d1]) / 400))
            
            # Actual score: d1 always beat d2 in adjacent comparison (1.0)
            actual_d1 = 1.0
            
            # Scale K-factor by position difference importance
            # Top positions matter more
            position_weight = 1.0 / (1 + (p1 / 10))  # Higher weight for top positions
            k_adjusted = ELO_K_FACTOR * position_weight
            
            # Update ratings - NO division by len(positions)
            elo[d1] += k_adjusted * (actual_d1 - expected_d1)
            elo[d2] += k_adjusted * ((1 - actual_d1) - (1 - expected_d1))
    
    print(f"  ✓ Calculated Elo for {len(elo)} drivers")
    
    # Print top 5
    sorted_elo = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5 Elo ratings:")
    for driver, rating in sorted_elo[:5]:
        print(f"    {driver}: {rating:.0f}")
    
    return elo


def calculate_recent_form(race_results: list, window: int = 5) -> dict:
    """
    Calculate recent form (average finish position over last N races)
    Lower is better.
    """
    print(f"\n📈 Calculating recent form (last {window} races)...")
    
    # Collect all finishes per driver
    driver_finishes = defaultdict(list)
    
    for race in race_results:
        for result in race.get("Results", []):
            driver_code = result.get("Driver", {}).get("code", "UNK")
            position = int(result.get("position", 20))
            driver_finishes[driver_code].append(position)
    
    # Calculate average of last N races
    recent_form = {}
    for driver, finishes in driver_finishes.items():
        recent = finishes[-window:] if len(finishes) >= window else finishes
        recent_form[driver] = sum(recent) / len(recent) if recent else 10.0
    
    print(f"  ✓ Calculated recent form for {len(recent_form)} drivers")
    
    # Print top 5
    sorted_form = sorted(recent_form.items(), key=lambda x: x[1])
    print("  Best recent form:")
    for driver, form in sorted_form[:5]:
        print(f"    {driver}: {form:.2f}")
    
    return recent_form


def calculate_circuit_history(race_results: list) -> dict:
    """
    Calculate average finish position per circuit per driver
    Uses proper circuit name normalization
    """
    print("\n🏁 Calculating circuit history...")
    
    # circuit -> driver -> [positions]
    circuit_driver_positions = defaultdict(lambda: defaultdict(list))
    
    for race in race_results:
        circuit_raw = race.get("Circuit", {}).get("circuitName", "Unknown")
        # Use proper circuit name mapping
        circuit = normalize_circuit_name(circuit_raw)
        
        for result in race.get("Results", []):
            driver_code = result.get("Driver", {}).get("code", "UNK")
            position = int(result.get("position", 20))
            circuit_driver_positions[circuit][driver_code].append(position)
    
    # Calculate averages
    circuit_history = {}
    for circuit, drivers in circuit_driver_positions.items():
        circuit_history[circuit] = {}
        for driver, positions in drivers.items():
            circuit_history[circuit][driver] = sum(positions) / len(positions)
    
    print(f"  ✓ Calculated history for {len(circuit_history)} circuits")
    
    return circuit_history


def calculate_team_performance(constructor_standings: list) -> dict:
    """
    Calculate team performance score based on constructor standings
    Normalized 0-1 (1 = best team)
    """
    print("\n🏎️ Calculating team performance scores...")
    
    team_scores = {}
    total_teams = len(constructor_standings)
    
    for i, constructor in enumerate(constructor_standings):
        team_name = constructor.get("Constructor", {}).get("name", "Unknown")
        # Normalize: 1st place = 1.0, last place = 0.0
        team_name = TEAM_MAPPINGS.get(team_name, team_name)
        team_scores[team_name] = (total_teams - i) / total_teams
    
    print(f"  ✓ Calculated scores for {len(team_scores)} teams")
    for team, score in team_scores.items():
        print(f"    {team}: {score:.2f}")
    
    return team_scores


def calculate_driver_experience(race_results: list) -> dict:
    """
    Calculate driver experience score based on races completed in 2025
    """
    print("\n👤 Calculating driver experience...")
    
    race_counts = defaultdict(int)
    
    for race in race_results:
        for result in race.get("Results", []):
            driver_code = result.get("Driver", {}).get("code", "UNK")
            race_counts[driver_code] += 1
    
    # Normalize by max races
    max_races = max(race_counts.values()) if race_counts else 1
    experience = {d: c / max_races for d, c in race_counts.items()}
    
    print(f"  ✓ Calculated experience for {len(experience)} drivers")
    
    return experience


# =============================================================================
# PARQUET UPDATE FUNCTIONS
# =============================================================================
def load_existing_parquet(filename: str) -> pd.DataFrame:
    """Load existing parquet file"""
    path = config.MODEL_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def update_driver_features_parquet(
    elo_ratings: dict,
    recent_form: dict,
    experience: dict,
    driver_standings: list,
    year: int
):
    """Update driver_features_snapshot.parquet with current year data"""
    print("\n💾 Updating driver features parquet...")
    
    # Load existing
    df = load_existing_parquet("driver_features_snapshot.parquet")
    
    # Create new rows for current year data
    new_rows = []
    
    for standing in driver_standings:
        driver_info = standing.get("Driver", {})
        constructor_info = standing.get("Constructors", [{}])[0]
        
        driver_code = driver_info.get("code", "UNK")
        driver_name = f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}"
        team = constructor_info.get("name", "Unknown")
        team = TEAM_MAPPINGS.get(team, team)
        
        new_rows.append({
            "driver": driver_code,
            "driver_name": driver_name,
            "team": team,
            "year": year,
            "EloRating": elo_ratings.get(driver_code, ELO_BASE),
            "RecentFormAvg": recent_form.get(driver_code, 10.0),
            "DriverExperienceScore": experience.get(driver_code, 0.5),
            "driver_enc": get_driver_encoding(driver_code),
            "team_enc": get_team_encoding(team)
        })
    
    # Validate data
    if len(new_rows) < 15:
        print(f"  ⚠️ Warning: Only found {len(new_rows)} drivers (expected ~20)")
    
    # Append new data
    new_df = pd.DataFrame(new_rows)
    
    # Remove old year data if exists, then append new
    if not df.empty and "year" in df.columns:
        df = df[df["year"] != year]
    
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    output_path = config.MODEL_DIR / "driver_features_snapshot.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    print(f"  Total rows: {len(df)}")
    
    return df


def update_circuit_features_parquet(circuit_history: dict, year: int):
    """Update circuit_features_snapshot.parquet with current year data"""
    print("\n💾 Updating circuit features parquet...")
    
    # Load existing
    df = load_existing_parquet("circuit_features_snapshot.parquet")
    
    # Create new rows
    new_rows = []
    
    for circuit, drivers in circuit_history.items():
        for driver, avg_pos in drivers.items():
            new_rows.append({
                "circuit": circuit,
                "driver": driver,
                "year": year,
                "CircuitHistoryAvg": avg_pos,
                "circuit_enc": get_circuit_encoding(circuit)
            })
    
    new_df = pd.DataFrame(new_rows)
    
    # Remove old year data
    if not df.empty and "year" in df.columns:
        df = df[df["year"] != year]
    
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    output_path = config.MODEL_DIR / "circuit_features_snapshot.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    
    return df


def update_team_features_parquet(team_scores: dict, year: int):
    """Update team_features_snapshot.parquet with current year data"""
    print("\n💾 Updating team features parquet...")
    
    if not team_scores:
        print("  ❌ Error: No team data to save!")
        return pd.DataFrame()
    
    # Load existing
    df = load_existing_parquet("team_features_snapshot.parquet")
    
    # Create new rows
    new_rows = []
    
    for team, score in team_scores.items():
        new_rows.append({
            "team": team,
            "year": year,
            "TeamPerfScore": score,
            "team_enc": get_team_encoding(team)
        })
    
    new_df = pd.DataFrame(new_rows)
    
    # Remove old year data
    if not df.empty and "year" in df.columns:
        df = df[df["year"] != year]
    
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    output_path = config.MODEL_DIR / "team_features_snapshot.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to {output_path}")
    
    return df


# =============================================================================
# SUPABASE UPDATE FUNCTIONS
# =============================================================================
def update_supabase_features(
    elo_ratings: dict,
    recent_form: dict,
    team_scores: dict,
    circuit_history: dict,
    experience: dict,
    year: int
):
    """Save feature data to Supabase"""
    print("\n☁️ Updating Supabase...")
    
    if not config.USE_SUPABASE:
        print("  ⚠️ Supabase not configured, skipping...")
        return
    
    try:
        from supabase import create_client
        
        supabase = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_SERVICE_KEY or config.SUPABASE_KEY
        )
        
        # Update drivers table with current features
        print("  Updating drivers...")
        driver_count = 0
        for driver, elo in elo_ratings.items():
            data = {
                "code": driver,
                "current_elo": float(elo),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            try:
                # Upsert - Supabase auto-detects conflict on unique key
                supabase.table("drivers").upsert(data).execute()
                driver_count += 1
            except Exception as e:
                print(f"    ⚠️ Failed to update {driver}: {e}")
        
        print(f"    ✓ Updated {driver_count} drivers")
        
        # Update teams table
        print("  Updating teams...")
        team_count = 0
        for team, score in team_scores.items():
            data = {
                "name": team,
                "team_perf_score": float(score),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            try:
                supabase.table("teams").upsert(data).execute()
                team_count += 1
            except Exception as e:
                print(f"    ⚠️ Failed to update {team}: {e}")
        
        print(f"    ✓ Updated {team_count} teams")
        print("  ✓ Supabase updated successfully")
        
    except ImportError:
        print("  ⚠️ Supabase package not installed. Run: pip install supabase")
    except Exception as e:
        print(f"  ❌ Supabase error: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main function to update all feature data"""
    print("=" * 60)
    print("F1 FEATURE UPDATE")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Use local variable instead of mutating global
    year = DEFAULT_YEAR
    print(f"Target Year: {year}")
    
    # 1. Fetch data from Ergast
    race_results = get_race_results(year)
    driver_standings = get_driver_standings(year)
    constructor_standings = get_constructor_standings(year)
    
    if not race_results:
        print(f"\n⚠️ No {year} race results found. Trying previous year...")
        year = year - 1
        race_results = get_race_results(year)
        driver_standings = get_driver_standings(year)
        constructor_standings = get_constructor_standings(year)
    
    if not race_results:
        print("\n❌ No data available. Exiting.")
        return
    
    # Validate data
    print(f"\n📋 Data Validation:")
    print(f"  Races found: {len(race_results)}")
    print(f"  Drivers in standings: {len(driver_standings)}")
    print(f"  Constructors in standings: {len(constructor_standings)}")
    
    if len(driver_standings) < 15:
        print(f"  ⚠️ Warning: Expected ~20 drivers, found {len(driver_standings)}")
    
    if not constructor_standings:
        print("  ❌ Error: No constructor standings found!")
        return
    
    # 2. Calculate features
    elo_ratings = calculate_elo_ratings(race_results)
    recent_form = calculate_recent_form(race_results)
    circuit_history = calculate_circuit_history(race_results)
    team_scores = calculate_team_performance(constructor_standings)
    experience = calculate_driver_experience(race_results)
    
    # 3. Update local parquet files
    print("\n" + "=" * 60)
    print("UPDATING LOCAL PARQUET FILES")
    print("=" * 60)
    
    update_driver_features_parquet(elo_ratings, recent_form, experience, driver_standings, year)
    update_circuit_features_parquet(circuit_history, year)
    update_team_features_parquet(team_scores, year)
    
    # 4. Update Supabase
    print("\n" + "=" * 60)
    print("UPDATING SUPABASE")
    print("=" * 60)
    
    update_supabase_features(
        elo_ratings, recent_form, team_scores, circuit_history, experience, year
    )
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("✅ UPDATE COMPLETE")
    print("=" * 60)
    print(f"  Year: {year}")
    print(f"  Races processed: {len(race_results)}")
    print(f"  Drivers updated: {len(elo_ratings)}")
    print(f"  Teams updated: {len(team_scores)}")
    print(f"  Circuits: {len(circuit_history)}")
    print("\nYour predictions will now use current data!")
    print("Restart the API server to load the new features.")


if __name__ == "__main__":
    main()
