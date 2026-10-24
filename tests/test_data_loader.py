import pytest
import pandas as pd
from utils.data_loader import load_data
from pathlib import Path

class TestDataLoader:
    @pytest.fixture(scope="session")
    def sample_path(self, tmp_path_factory):
        # Create test data
        tmp_dir = tmp_path_factory.mktemp("data")
        test_file = tmp_dir / "sample_train.csv"
        
        df = pd.DataFrame({
            'PremiumAmount': [100, 200, 300],
            'Age': [25, 35, 45],
            'Coverage': [50000, 75000, 100000]
        })
        df.to_csv(test_file, index=False)
        return test_file

    def test_load_data_success(self, sample_path):
        df = load_data(sample_path)
        assert not df.empty
        assert 'PremiumAmount' in df.columns
        assert len(df) == 3

    def test_load_data_failure(self):
        with pytest.raises(FileNotFoundError):
            load_data(Path("non_existent.csv"))

    def test_load_data_none_path(self):
        with pytest.raises(ValueError):
            load_data(None)

    def test_load_data_str_path(self, sample_path):
        df = load_data(str(sample_path))
        assert isinstance(df, pd.DataFrame)