"""
Unit Tests for Stage 3: Prompt Generation
"""

import pytest
import numpy as np
from src.stage3_prompt import PromptGenerator, StyleTemplates, SceneAnalyzer


class TestStyleTemplates:
    """Test style templates"""
    
    def test_get_all_styles(self):
        styles = StyleTemplates.get_all_styles()
        assert len(styles) == 3
        assert "modern" in styles
    
    def test_get_style_prompt(self):
        prompt = StyleTemplates.get_style_prompt("modern", "bedroom")
        assert "bedroom" in prompt.lower()
        assert len(prompt) > 50


class TestSceneAnalyzer:
    """Test scene analyzer"""
    
    def test_bedroom_classification(self):
        analyzer = SceneAnalyzer()
        objects = ["bed", "chair", "pottedplant"]
        result = analyzer.analyze(objects)
        assert result['room_type'] == "bedroom"
    
    def test_living_room_classification(self):
        analyzer = SceneAnalyzer()
        objects = ["sofa", "tvmonitor", "chair"]
        result = analyzer.analyze(objects)
        assert result['room_type'] == "living_room"
    
    def test_with_stage2_room_type_hint(self):
        analyzer = SceneAnalyzer()
        objects = ["bed"]
        result = analyzer.analyze(objects, room_type_hint="Bedroom")
        assert result['room_type'] == "bedroom"
    
    def test_brightness_analysis(self):
        analyzer = SceneAnalyzer()
        
        dark_image = np.ones((512, 512, 3), dtype=np.uint8) * 50
        result = analyzer.analyze([], dark_image)
        assert result['brightness'] == "dark"
        
        bright_image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        result = analyzer.analyze([], bright_image)
        assert result['brightness'] == "bright"
    
    def test_room_size_from_depth(self):  # FIXED: INSIDE CLASS NOW
        """Test size estimation from depth map"""
        analyzer = SceneAnalyzer()
        
        small_depth = np.ones((512, 512)) * 50 + np.random.rand(512, 512) * 5
        result = analyzer.analyze([], depth_map=small_depth)
        assert result['room_size'] == "small"
        
        large_depth = np.random.rand(512, 512) * 1000
        result = analyzer.analyze([], depth_map=large_depth)
        assert result['room_size'] == "large"


class TestPromptGenerator:
    """Test main prompt generator"""
    
    def test_generic_mode(self):
        gen = PromptGenerator()
        result = gen.generate(
            mode="generic",
            detected_objects=["bed", "chair"],
            style="modern"
        )
        assert result['mode'] == "generic"
        assert result['style'] == "modern"
        assert 'prompt' in result
    
    def test_prompt_mode(self):
        gen = PromptGenerator()
        custom = "Rustic cabin bedroom with wooden furniture"
        result = gen.generate(
            mode="prompt",
            detected_objects=["bed"],
            custom_prompt=custom
        )
        assert result['mode'] == "prompt"
        assert custom in result['prompt']
    
    def test_auto_mode(self):  # FIXED: ADDED depth_map
        """Test auto-design mode"""
        gen = PromptGenerator()
        result = gen.generate(
            mode="auto",
            detected_objects=["bed", "chair", "pottedplant"],
            image_array=np.ones((512, 512, 3), dtype=np.uint8) * 150,
            depth_map=np.random.rand(512, 512) * 200
        )
        assert result['mode'] == "auto"
        assert 'reasoning' in result['metadata']
        assert len(result['prompt']) > 0
    
    def test_integration_with_stage2_output(self):  # FIXED: ADDED depth_map
        """Test with actual Stage 2 format"""
        gen = PromptGenerator()
        stage2_objects = ["bed", "chair", "pottedplant"]
        
        result = gen.generate(
            mode="auto",
            detected_objects=stage2_objects,
            image_array=np.ones((512, 512, 3), dtype=np.uint8) * 180,
            depth_map=np.random.rand(512, 512) * 200
        )
        assert result['metadata']['room_type'] == "bedroom"
        assert "bed" in result['metadata']['objects']


class TestStage3Integration:
    """Integration tests"""
    
    def test_full_stage2_to_stage3_flow(self):
        """Test complete Stage 2 → Stage 3 flow"""
        stage2_inventory = {
            "room_type": "Bedroom",
            "detected_objects": ["bed", "chair"]
        }
        
        stage1_image = np.ones((512, 512, 3), dtype=np.uint8) * 160
        stage1_depth = np.random.rand(512, 512) * 200
        
        generator = PromptGenerator()
        result = generator.generate(
            mode="auto",
            detected_objects=stage2_inventory['detected_objects'],
            image_array=stage1_image,
            depth_map=stage1_depth
        )
        
        assert 'prompt' in result
        assert 'negative_prompt' in result
        assert isinstance(result['prompt'], str)
        assert len(result['prompt']) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])