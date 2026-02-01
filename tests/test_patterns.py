"""Tests for weight key pattern matching."""

import pytest
from modelopt_streaming.patterns import should_quantize_tensor


class TestPatternMatching:
    """Test suite for should_quantize_tensor()."""
    
    def test_mlp_only_standard(self):
        """Test MLP-only mode with standard MLP layers."""
        assert should_quantize_tensor("model.layers.0.mlp.down_proj.weight", mlp_only=True)
        assert should_quantize_tensor("model.layers.0.mlp.gate_proj.weight", mlp_only=True)
        assert should_quantize_tensor("model.layers.0.mlp.up_proj.weight", mlp_only=True)
    
    def test_mlp_only_moe_experts(self):
        """Test MLP-only mode with MoE expert layers."""
        assert should_quantize_tensor("model.layers.1.mlp.experts.0.down_proj.weight", mlp_only=True)
        assert should_quantize_tensor("model.layers.1.mlp.experts.127.gate_proj.weight", mlp_only=True)
    
    def test_mlp_only_shared_experts(self):
        """Test MLP-only mode with shared expert layers."""
        assert should_quantize_tensor("model.layers.1.mlp.shared_experts.down_proj.weight", mlp_only=True)
    
    def test_mlp_only_excludes_attention(self):
        """Test MLP-only mode excludes attention weights."""
        assert not should_quantize_tensor("model.layers.0.self_attn.q_proj.weight", mlp_only=True)
        assert not should_quantize_tensor("model.layers.0.self_attn.k_proj.weight", mlp_only=True)
        assert not should_quantize_tensor("model.layers.0.self_attn.o_proj.weight", mlp_only=True)
    
    def test_mlp_only_excludes_embeddings(self):
        """Test MLP-only mode excludes embeddings and lm_head."""
        assert not should_quantize_tensor("model.embed_tokens.weight", mlp_only=True)
        assert not should_quantize_tensor("lm_head.weight", mlp_only=True)
    
    def test_mlp_only_excludes_norms(self):
        """Test MLP-only mode excludes normalization layers."""
        assert not should_quantize_tensor("model.layers.0.input_layernorm.weight", mlp_only=True)
        assert not should_quantize_tensor("model.norm.weight", mlp_only=True)
    
    def test_all_linear_includes_attention(self):
        """Test all-linear mode includes attention weights."""
        assert should_quantize_tensor("model.layers.0.self_attn.q_proj.weight", mlp_only=False)
        assert should_quantize_tensor("model.layers.0.self_attn.k_proj.weight", mlp_only=False)
    
    def test_all_linear_includes_mlp(self):
        """Test all-linear mode includes MLP weights."""
        assert should_quantize_tensor("model.layers.0.mlp.down_proj.weight", mlp_only=False)
    
    def test_all_linear_excludes_embeddings(self):
        """Test all-linear mode still excludes embeddings."""
        assert not should_quantize_tensor("model.embed_tokens.weight", mlp_only=False)
        assert not should_quantize_tensor("lm_head.weight", mlp_only=False)
    
    def test_ignores_non_weight_tensors(self):
        """Test that non-weight tensors are never quantized."""
        assert not should_quantize_tensor("model.layers.0.mlp.down_proj.bias", mlp_only=True)
        assert not should_quantize_tensor("model.layers.0.input_layernorm.weight", mlp_only=True)
