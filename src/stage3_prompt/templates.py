"""
Style Templates for Generic Mode
Author: [Your Name] - Stage 3 Developer
Purpose: Predefined prompts for Modern, Minimal, Luxury styles
"""

class StyleTemplates:
    """
    Stores predefined style templates for quick styling
    
    Why this class?
    - Separates data from logic (clean code principle)
    - Easy to add new styles later
    - Reusable across the project
    """
    
    # Base quality modifiers (used in all prompts)
    QUALITY_SUFFIX = ", high quality, detailed, professional interior design, 8k resolution"
    
    # Negative prompt (what we DON'T want)
    NEGATIVE_PROMPT = "blurry, distorted, low quality, unrealistic, cluttered, messy"
    
    # Generic Mode Templates
    GENERIC_STYLES = {
        "modern": {
            "name": "Modern",
            "description": "Clean lines, minimalist furniture, neutral colors",
            "prompt_template": (
                "Modern {room_type} interior design with clean lines, "
                "minimalist furniture, neutral color palette with white and gray tones, "
                "contemporary style, geometric shapes, sleek finishes"
            ),
        },
        
        "minimal": {
            "name": "Minimal",
            "description": "Essential furniture only, maximum simplicity",
            "prompt_template": (
                "Minimalist {room_type} with essential furniture only, "
                "white walls, simple clean design, uncluttered space, "
                "Scandinavian style, natural light, wooden accents"
            ),
        },
        
        "luxury": {
            "name": "Luxury",
            "description": "Premium materials, elegant decor, rich colors",
            "prompt_template": (
                "Luxurious {room_type} interior with premium materials, "
                "elegant furniture, rich color palette with gold accents, "
                "velvet textures, marble surfaces, chandelier lighting, "
                "opulent and sophisticated design"
            ),
        },
    }
    
    # Room-specific enhancements
    ROOM_ENHANCEMENTS = {
        "bedroom": "comfortable bed, soft lighting, relaxing atmosphere",
        "living_room": "comfortable seating, entertainment area, welcoming ambiance",
        "kitchen": "modern appliances, clean countertops, functional layout",
        "bathroom": "clean tiles, modern fixtures, spa-like atmosphere",
        "office": "ergonomic desk, organized shelves, productive environment",
    }
    
    @classmethod
    def get_style_prompt(cls, style_name: str, room_type: str = "room") -> str:
        """
        Get complete prompt for a generic style
        
        Args:
            style_name: "modern", "minimal", or "luxury"
            room_type: "bedroom", "living_room", etc.
        
        Returns:
            Complete prompt string ready for Stable Diffusion
        
        Example:
            >>> StyleTemplates.get_style_prompt("modern", "bedroom")
            "Modern bedroom interior design with clean lines..."
        """
        # Get the style template
        if style_name.lower() not in cls.GENERIC_STYLES:
            raise ValueError(f"Style '{style_name}' not found. Choose: modern, minimal, luxury")
        
        style = cls.GENERIC_STYLES[style_name.lower()]
        
        # Format the template with room type
        base_prompt = style["prompt_template"].format(room_type=room_type)
        
        # Add room-specific enhancements if available
        room_enhancement = cls.ROOM_ENHANCEMENTS.get(room_type.lower(), "")
        if room_enhancement:
            base_prompt += f", {room_enhancement}"
        
        # Add quality suffix
        full_prompt = base_prompt + cls.QUALITY_SUFFIX
        
        return full_prompt
    
    @classmethod
    def get_all_styles(cls) -> list:
        """
        Get list of available style names
        
        Returns:
            List of style names: ["modern", "minimal", "luxury"]
        """
        return list(cls.GENERIC_STYLES.keys())
    
    @classmethod
    def get_style_info(cls, style_name: str) -> dict:
        """
        Get detailed information about a style
        
        Args:
            style_name: Style to get info for
        
        Returns:
            Dictionary with name, description, and template
        """
        if style_name.lower() not in cls.GENERIC_STYLES:
            return None
        return cls.GENERIC_STYLES[style_name.lower()]


# Example usage and testing
if __name__ == "__main__":
    # Test the templates
    print("=== Testing Style Templates ===\n")
    
    # Test 1: Get all available styles
    print("Available styles:", StyleTemplates.get_all_styles())
    print()
    
    # Test 2: Generate prompts for each style
    for style in ["modern", "minimal", "luxury"]:
        prompt = StyleTemplates.get_style_prompt(style, "bedroom")
        print(f"{style.upper()} BEDROOM:")
        print(prompt)
        print("-" * 80)
        print()
    
    # Test 3: Different room types
    print("MODERN LIVING ROOM:")
    print(StyleTemplates.get_style_prompt("modern", "living_room"))
    print()
    
    # Test 4: Get style info
    info = StyleTemplates.get_style_info("luxury")
    print("LUXURY STYLE INFO:")
    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")