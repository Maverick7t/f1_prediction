from supabase import create_client
import os, json
from dotenv import load_dotenv
load_dotenv('.env')
sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))
res = sb.table('predictions').select('id', count='exact').eq('race_year', 2025).execute()
print('Total 2025 predictions in DB:', res.count)
races = sb.table('predictions').select('race,predicted,actual,correct,confidence').eq('race_year', 2025).neq('race', '__SEASON_SUMMARY_2025__').order('timestamp').execute()
for r in races.data:
    mark = 'OK' if r['correct'] else 'XX'
    race_name = r['race']
    pred = r['predicted']
    actual = r['actual']
    conf = r['confidence']
    print(f"  {mark} {race_name:35s} Pred={pred:5s} Actual={actual:5s} Conf={conf}")

summary = sb.table('predictions').select('full_predictions').eq('race', '__SEASON_SUMMARY_2025__').execute()
if summary.data:
    stats = json.loads(summary.data[0]['full_predictions'])['stats']
    print(f"\nSummary: {stats['correct_predictions']}/{stats['total_races']} correct")
    print(f"  Winner accuracy: {stats['accuracy_percentage']}%")
    print(f"  Podium accuracy: {stats['podium_accuracy_percentage']}%")
