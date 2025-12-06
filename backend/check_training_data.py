#!/usr/bin/env python3
"""Check training data"""
import pandas as pd
import os

csv_file = 'f1_training_dataset_2018_2024.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print('=== TRAINING DATA ===')
    print(f'Shape: {df.shape}')
    print(f'Years: {sorted(df["race_year"].unique())}')
    print(f'Columns: {list(df.columns)}')
    print(f'\nSample data (2024):')
    sample = df[df['race_year'] == 2024][['race_year', 'driver', 'team', 'event', 'finishing_position']].head(3)
    print(sample.to_string())
else:
    print(f'File not found: {csv_file}')
