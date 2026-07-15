import pytest
import os
from unittest.mock import patch
from vectorstore.store import FaissStore
import vectorstore.embedder as embedder

def test_supabase_pull_failure_does_not_crash(tmp_path):
    """
    Test that if pull_from_supabase raises a network error during 
    FaissStore initialization or embedder cache loading, the application 
    catches it gracefully and does not crash.
    """
    with patch("vectorstore.store.pull_from_supabase", side_effect=Exception("Mock pull network error")) as mock_store_pull:
        with patch("storage.remote_sync.pull_from_supabase", side_effect=Exception("Mock pull network error")) as mock_embedder_pull:
            
            # 1. Test FaissStore initialization (should not crash)
            data_dir = tmp_path / "data"
            store = FaissStore(str(data_dir))
            
            # Verify pull was attempted multiple times
            assert mock_store_pull.call_count >= 4
            
            # 2. Test embedder cache loading (should not crash)
            embedder._load_cache()
            assert mock_embedder_pull.call_count >= 1

def test_supabase_push_failure_does_not_crash(tmp_path):
    """
    Test that if push_to_supabase raises a network error during 
    saving operations, the application catches it gracefully and 
    does not crash.
    """
    with patch("storage.remote_sync.push_to_supabase", side_effect=Exception("Mock push network error")) as mock_push:
        
        # 1. Test FaissStore save (should not crash)
        data_dir = tmp_path / "data"
        store = FaissStore(str(data_dir))
        store.save()
        
        # Verify push was attempted multiple times
        assert mock_push.call_count >= 4
        
        # 2. Test embedder cache saving (should not crash)
        embedder._save_cache()
        assert mock_push.call_count >= 5
