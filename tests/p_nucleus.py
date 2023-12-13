import pytest
import torch
from predict import top_p_filtering  # Import your top_p_filtering function

# Fixture for creating mock logits
@pytest.fixture
def mock_logits():
    return torch.randn(10)  # A tensor with random logits

# Test for top_p_filtering with top_p = 0.9
def test_top_p_filtering_90(mock_logits):
    filtered_logits = top_p_filtering(mock_logits, top_p=0.9)
    assert torch.is_tensor(filtered_logits), "Output should be a tensor"
    assert not torch.any(filtered_logits == -float('Inf')), "No logits should be set to -Inf with top_p=0.9"

# Test for top_p_filtering with top_p = 0.5
def test_top_p_filtering_50(mock_logits):
    filtered_logits = top_p_filtering(mock_logits, top_p=0.5)
    assert torch.is_tensor(filtered_logits), "Output should be a tensor"
    assert torch.any(filtered_logits == -float('Inf')), "Some logits should be set to -Inf with top_p=0.5"

