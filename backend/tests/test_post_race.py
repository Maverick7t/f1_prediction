import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.scripts import post_race


class PostRaceTests(unittest.TestCase):
    @patch.object(post_race, "fetch_winner_code", return_value="ANT")
    @patch.object(post_race, "fetch_results", return_value=[{"driver": "Antonelli", "team": "Mercedes"}])
    @patch.object(
        post_race,
        "fetch_race_meta",
        return_value=SimpleNamespace(
            race_key="2026_10_Belgian_Grand_Prix",
            round=10,
            race_name="Belgian Grand Prix",
            circuit_name="Circuit de Spa-Francorchamps",
            circuit_id="spa",
            date="2026-07-19",
        ),
    )
    @patch.object(post_race, "ensure_directories")
    @patch.object(post_race.config, "SUPABASE_URL", "https://example.supabase.co")
    @patch.object(post_race.config, "SUPABASE_SERVICE_KEY", "service-key")
    @patch.object(post_race, "get_prediction_logger")
    @patch.object(post_race, "PipelineStore")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_returns_zero_when_persistence_fails(
        self,
        mock_parse_args,
        mock_pipeline_store,
        _mock_prediction_logger,
        _mock_ensure_dirs,
        _mock_fetch_meta,
        _mock_fetch_results,
        _mock_fetch_winner,
    ):
        mock_parse_args.return_value = SimpleNamespace(
            year=2026,
            round_number=10,
            auto=False,
            dry_run=False,
            retrain=True,
        )
        mock_store = Mock()
        mock_store.upsert_results_raw.side_effect = RuntimeError("dns resolution failed")
        mock_pipeline_store.return_value = mock_store

        result = post_race.main()

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
