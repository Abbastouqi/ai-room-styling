"""
Prompt Generator - Main Logic
Author: [Your Name] - Stage 3 Developer
Purpose: Generate prompts for Stable Diffusion in 3 modes
"""

from typing import Dict, Optional, List
import numpy as np

from .templates import StyleTemplates
from .analyzer import SceneAnalyzer


class PromptGenerator:
    """
    Main Prompt Generation System
    
    Supports 3 modes:
    1. Generic Mode: User picks style (modern/minimal/luxury)
    2. Prompt Mode: User provides custom prompt
    3. Auto Mode: AI decides based on scene analysis
    """
    
    def __init__(self):
        """Initialize generator with analyzer and templates"""
        self.analyzer = SceneAnalyzer()
        self.templates = StyleTemplates()
        self.last_prompt = None
        self.last_analysis = None
    
    def generate(self,
                 mode: str,
                 detected_objects: List[str],
                 image_array: Optional[np.ndarray] = None,
                 depth_map: Optional[np.ndarray] = None,
                 style: Optional[str] = None,
                 custom_prompt: Optional[str] = None) -> Dict:
        """
        Main generation function
        
        Args:
            mode: "generic", "prompt", or "auto"
            detected_objects: From Stage 2 (YOLO)
            image_array: From Stage 1 (for brightness)
            depth_map: From Stage 1 (MiDaS - for size)
            style: For generic mode
            custom_prompt: For prompt mode
        """
        analysis = self.analyzer.analyze(detected_objects, image_array, depth_map)
        self.last_analysis = analysis
        
        if mode.lower() == "generic":
            return self._generate_generic(analysis, style)
        elif mode.lower() == "prompt":
            return self._generate_custom(analysis, custom_prompt)
        elif mode.lower() == "auto":
            return self._generate_auto(analysis)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _generate_generic(self, analysis: Dict, style: str) -> Dict:
        """MODE 1: Generic Style Mode"""
        if not style:
            raise ValueError("Style required for generic mode")
        
        room_type = analysis['room_type']
        prompt = self.templates.get_style_prompt(style, room_type)
        self.last_prompt = prompt
        
        return {
            'prompt': prompt,
            'negative_prompt': self.templates.NEGATIVE_PROMPT,
            'mode': 'generic',
            'style': style,
            'metadata': {
                'room_type': room_type,
                'objects': analysis['objects'],
                'scene_analysis': analysis
            }
        }

    def _generate_custom(self, analysis: Dict, custom_prompt: str) -> Dict:
        """MODE 2: Prompt-Based Mode"""
        if not custom_prompt:
            raise ValueError("Custom prompt required")
        
        room_type = analysis['room_type']
        
        if room_type not in custom_prompt.lower():
            enhanced_prompt = f"{custom_prompt}, {room_type} interior"
        else:
            enhanced_prompt = custom_prompt
        
        final_prompt = enhanced_prompt + self.templates.QUALITY_SUFFIX
        self.last_prompt = final_prompt
        
        return {
            'prompt': final_prompt,
            'negative_prompt': self.templates.NEGATIVE_PROMPT,
            'mode': 'prompt',
            'original_prompt': custom_prompt,
            'metadata': {
                'room_type': room_type,
                'objects': analysis['objects'],
                'scene_analysis': analysis
            }
        }

    def _generate_auto(self, analysis: Dict) -> Dict:
        """MODE 3: AI Auto-Design Mode"""
        room_type = analysis['room_type']
        brightness = analysis['brightness']
        room_size = analysis['room_size']
        is_cluttered = analysis['is_cluttered']
        object_count = analysis['object_count']
        
        prompt_parts = []
        prompt_parts.append(f"Design a {room_type} interior")
        
        if brightness == "dark":
            prompt_parts.append("with warm ambient lighting, cozy atmosphere")
        elif brightness == "bright":
            prompt_parts.append("with natural well-lit atmosphere")
        else:
            prompt_parts.append("with balanced comfortable lighting")
        
        if room_size == "small":
            prompt_parts.append("space-efficient compact design")
        elif room_size == "large":
            prompt_parts.append("spacious open layout with grand proportions")
        else:
            prompt_parts.append("well-proportioned comfortable space")
        
        if is_cluttered:
            prompt_parts.append("clean and organized")
        else:
            prompt_parts.append("minimal and uncluttered")
        
        prompt_parts.append("modern contemporary style with neutral tones")
        
        auto_prompt = ", ".join(prompt_parts) + self.templates.QUALITY_SUFFIX
        example_prompt = f"Design a clean, well-lit modern {room_type} with neutral tones"
        
        self.last_prompt = auto_prompt
        
        return {
            'prompt': auto_prompt,
            'negative_prompt': self.templates.NEGATIVE_PROMPT,
            'mode': 'auto',
            'example_format': example_prompt,
            'metadata': {
                'room_type': room_type,
                'brightness': brightness,
                'room_size': room_size,
                'is_cluttered': is_cluttered,
                'object_count': object_count,
                'objects': analysis['objects'],
                'reasoning': self._explain_auto_decision(analysis)
            }
        }
    
    def _explain_auto_decision(self, analysis: Dict) -> str:
        """
        Explain why auto mode made certain choices
        
        This is useful for debugging and user transparency
        """
        explanations = []
        
        if analysis['brightness'] == "dark":
            explanations.append("Room is dark → Added warm lighting")
        elif analysis['brightness'] == "bright":
            explanations.append("Room is bright → Emphasized natural light")
        
        if analysis['is_cluttered']:
            explanations.append("Room is cluttered → Suggested minimal design")
        else:
            explanations.append("Room is minimal → Enhanced clean aesthetic")
        
        if analysis['room_type'] != "room":
            explanations.append(f"Detected {analysis['room_type']} → Added specific enhancements")
        
        return " | ".join(explanations)
    
    def get_last_analysis(self) -> Optional[Dict]:
        """Get the last scene analysis"""
        return self.last_analysis
    
    def get_last_prompt(self) -> Optional[str]:
        """Get the last generated prompt"""
        return self.last_prompt


# Testing
if __name__ == "__main__":
    print("=== Testing Prompt Generator ===\n")
    
    generator = PromptGenerator()
    
    # Test data
    bedroom_objects = ["bed", "lamp", "nightstand", "window"]
    bright_image = np.ones((512, 512, 3)) * 200
    
    # Test 1: Generic Mode
    print("TEST 1: GENERIC MODE (Modern)")
    print("="*80)
    result1 = generator.generate(
        mode="generic",
        detected_objects=bedroom_objects,
        image_array=bright_image,
        style="modern"
    )
    print(f"Prompt: {result1['prompt']}")
    print(f"\nNegative: {result1['negative_prompt']}")
    print(f"\nMetadata: {result1['metadata']}")
    print("\n" + "="*80 + "\n")
    
    # Test 2: Custom Prompt Mode
    print("TEST 2: CUSTOM PROMPT MODE")
    print("="*80)
    result2 = generator.generate(
        mode="prompt",
        detected_objects=bedroom_objects,
        image_array=bright_image,
        custom_prompt="Cozy rustic cabin style bedroom with wooden furniture"
    )
    print(f"Original: {result2['original_prompt']}")
    print(f"\nEnhanced: {result2['prompt']}")
    print("\n" + "="*80 + "\n")
    
    # Test 3: Auto Mode
    print("TEST 3: AUTO MODE")
    print("="*80)
    result3 = generator.generate(
        mode="auto",
        detected_objects=bedroom_objects,
        image_array=bright_image
    )
    print(f"Prompt: {result3['prompt']}")
    print(f"\nReasoning: {result3['metadata']['reasoning']}")
    print(f"\nScene Analysis:")
    print(f"  - Room Type: {result3['metadata']['room_type']}")
    print(f"  - Brightness: {result3['metadata']['brightness']}")
    print(f"  - Cluttered: {result3['metadata']['is_cluttered']}")
    print("\n" + "="*80 + "\n")
    
    # Test 4: Dark cluttered living room (Auto mode)
    print("TEST 4: AUTO MODE - Dark Cluttered Living Room")
    print("="*80)
    living_objects = ["sofa", "tv", "table", "chair", "lamp", "plant", 
                      "bookshelf", "rug", "picture", "vase", "cushion"]
    dark_image = np.ones((512, 512, 3)) * 60
    
    result4 = generator.generate(
        mode="auto",
        detected_objects=living_objects,
        image_array=dark_image
    )
    print(f"Prompt: {result4['prompt']}")
    print(f"\nReasoning: {result4['metadata']['reasoning']}")
    print(f"\nObjects detected: {result4['metadata']['object_count']}")