import unittest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openevolve.database import ProgramDatabase, Program
from openevolve.config import Config, DatabaseConfig

class TestPTIS(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            database=DatabaseConfig(
                num_islands=2,
                pool_temperatures=[1.0, 10.0],
                lambda_reweight=1.0,
                gamma_exponent=0.5,
                migration_swap_rate=1.0,
                feature_dimensions=["score"]
            )
        )
        self.db = ProgramDatabase(self.config)
        
        # Add some initial programs
        p1 = Program(id="p1", code="print(1)", metrics={"combined_score": 0.1})
        p2 = Program(id="p2", code="print(2)", metrics={"combined_score": 0.9})
        self.db.add(p1, target_island=0)
        self.db.add(p2, target_island=0)
        self.db.add(p1, target_island=1)
        self.db.add(p2, target_island=1)
        self.db.island_best_programs[0] = "p2"
        self.db.island_best_programs[1] = "p1"

    def test_is_sampling(self):
        # test _sample_from_island_is
        parent, weight = self.db._sample_from_island_is(0)
        self.assertIn(parent.id, ["p1", "p2"])
        self.assertGreater(weight, 0)
        
    def test_is_correction(self):
        # test correction in add()
        p_new = Program(id="p_new", code="print(new)", metrics={"combined_score": 0.5})
        self.db.add(p_new, importance_weight=2.0)
        # An importance weight greater than one increases the adjusted score.
        self.assertGreater(p_new.metrics["combined_score"], 0.5)

    def test_pt_swap_acceptance(self):
        # Moving the better program into the cold pool has positive log acceptance.
        
        self.db.island_best_programs[0] = "p1" # Worst to cold
        self.db.island_best_programs[1] = "p2" # Best to hot
        
        with self.assertLogs('openevolve.database', level='INFO') as cm:
            self.db._perform_pt_swaps()
            self.assertTrue(any("PT Swap ACCEPTED" in line for line in cm.output))

if __name__ == "__main__":
    unittest.main()
