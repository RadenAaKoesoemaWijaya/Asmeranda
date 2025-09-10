<<<<<<< HEAD
import random
import string
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class CaptchaGenerator:
    def __init__(self):
        self.width = 200
        self.height = 80
        self.length = 5
        self.font_size = 28
    
    def generate_text(self):
        """Generate random captcha text"""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(self.length))
    
    def generate_captcha(self, text=None):
        """Generate captcha image"""
        if text is None:
            text = self.generate_text()
        
        # Create image with white background
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Add some noise
        for _ in range(100):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.point((x, y), fill='black')
        
        # Add some lines
        for _ in range(5):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            draw.line([(x1, y1), (x2, y2)], fill='gray', width=1)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype('arial.ttf', self.font_size)
        except:
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', self.font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        text_width = draw.textlength(text, font=font)
        x = (self.width - text_width) // 2
        y = (self.height - self.font_size) // 2
        
        # Random color for text
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        draw.text((x, y), text, font=font, fill=color)
        
        # Add some distortion
        for _ in range(3):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.arc([(x, y), (x+30, y+30)], 0, 360, fill='gray')
        
        return image, text
    
    def get_captcha_base64(self, text=None):
        """Generate captcha and return as base64 string"""
        image, actual_text = self.generate_captcha(text)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}", actual_text

def verify_captcha(input_text, actual_text, case_sensitive=True):
    """Verify captcha input"""
    if case_sensitive:
        return input_text.strip() == actual_text.strip()
    else:
        return input_text.strip().upper() == actual_text.strip().upper()

# Initialize global captcha generator
=======
import random
import string
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class CaptchaGenerator:
    def __init__(self):
        self.width = 200
        self.height = 80
        self.length = 5
        self.font_size = 28
    
    def generate_text(self):
        """Generate random captcha text"""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(self.length))
    
    def generate_captcha(self, text=None):
        """Generate captcha image"""
        if text is None:
            text = self.generate_text()
        
        # Create image with white background
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Add some noise
        for _ in range(100):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.point((x, y), fill='black')
        
        # Add some lines
        for _ in range(5):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            draw.line([(x1, y1), (x2, y2)], fill='gray', width=1)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype('arial.ttf', self.font_size)
        except:
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', self.font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        text_width = draw.textlength(text, font=font)
        x = (self.width - text_width) // 2
        y = (self.height - self.font_size) // 2
        
        # Random color for text
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        draw.text((x, y), text, font=font, fill=color)
        
        # Add some distortion
        for _ in range(3):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.arc([(x, y), (x+30, y+30)], 0, 360, fill='gray')
        
        return image, text
    
    def get_captcha_base64(self, text=None):
        """Generate captcha and return as base64 string"""
        image, actual_text = self.generate_captcha(text)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}", actual_text

def verify_captcha(input_text, actual_text, case_sensitive=True):
    """Verify captcha input"""
    if case_sensitive:
        return input_text.strip() == actual_text.strip()
    else:
        return input_text.strip().upper() == actual_text.strip().upper()

# Initialize global captcha generator
>>>>>>> 82a9a7837d554aa663a3debb5b0cd475375882e8
captcha_gen = CaptchaGenerator()