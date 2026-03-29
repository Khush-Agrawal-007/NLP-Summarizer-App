# app/services/technical_summarizer.py
"""
Technical Product Summarizer using LangChain for structured output
"""

from typing import Dict, List, Any, Optional
from langchain_core.prompts import PromptTemplate
from app.utils.model_utils import load_model_tokenizer, generate_summary
import torch


class TechnicalSummarizer:
    """
    Specialized summarizer for technical products with structured output
    Uses LangChain for prompt engineering and transformers for generation
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        print(f"Loading technical summarizer model: {model_name}")
        try:
            self.model, self.tokenizer, self.device = load_model_tokenizer(model_name)
            # Define structured output prompt
            self.structured_prompt = self._create_structured_prompt()
            print("Technical summarizer initialized successfully")
        except Exception as e:
            print(f"Error initializing technical summarizer: {e}")
            raise
    
    def _create_structured_prompt(self) -> PromptTemplate:
        """
        Create a prompt template for structured product summarization
        """
        template = """You are a technical product analyst. Extract and summarize the following product description into a structured format.

Product Description:
{product_description}

Extract the following information:
1. Product name and category
2. Key technical specifications
3. Pros (advantages)
4. Cons (disadvantages)
5. Best use case
6. Price range

Provide a concise, technical summary focusing on specifications and performance characteristics.
"""
        return PromptTemplate(
            input_variables=["product_description"],
            template=template
        )
    
    def _safe_summarize(self, text: str, max_length: int = 400, min_length: int = 200) -> str:
        """
        Safely summarize text using the model directly
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summarized text
        """
        try:
            return generate_summary(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                device=self.device,
                max_length=max_length,
                min_length=min_length,
            )
            
        except Exception as e:
            print(f"Error in safe summarization: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def summarize_to_text(
        self, 
        product_description: str, 
        max_length: int = 200,
        min_length: int = 50
    ) -> str:
        """
        Generate a text summary of a product description
        
        Args:
            product_description: Full product description
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Text summary
        """
        if not product_description or len(product_description.strip()) < 50:
            return "Input text is too short. Please provide a more detailed product description."
        
        try:
            # Check text length and handle accordingly
            tokens = self.tokenizer.encode(product_description, add_special_tokens=False, max_length=1024, truncation=True)
            
            # Use safe summarization method
            summary = self._safe_summarize(product_description, max_length=max_length, min_length=min_length)
            
            if summary:
                return summary
            else:
                return "Unable to generate summary. Please try with a different product description."
                
        except Exception as e:
            print(f"Error in summarize_to_text: {e}")
            return "An error occurred while generating the summary. Please ensure the input text is valid and try again."
    
    def extract_structured_summary(self, product_description: str) -> Dict[str, Any]:
        """
        Extract structured information from product description
        Uses rule-based extraction combined with summarization
        
        Args:
            product_description: Full product description
            
        Returns:
            Dictionary with structured product information
        """
        # Generate base summary
        base_summary = self.summarize_to_text(product_description, max_length=250)
        
        # Extract key specifications using keyword matching
        key_specs = self._extract_specifications(product_description)
        
        # Extract pros and cons
        pros, cons = self._extract_pros_cons(product_description)
        
        # Determine product category
        category = self._determine_category(product_description)
        
        # Extract product name (usually first proper noun or mentioned brand)
        product_name = self._extract_product_name(product_description)
        
        # Determine best use case
        best_for = self._determine_use_case(product_description, base_summary)
        
        # Estimate price range
        price_range = self._estimate_price_range(product_description)
        
        return {
            "product_name": product_name,
            "category": category,
            "summary": base_summary,
            "key_specs": key_specs,
            "pros": pros,
            "cons": cons,
            "best_for": best_for,
            "price_range": price_range
        }
    
    def _extract_specifications(self, text: str) -> Dict[str, str]:
        """Extract technical specifications from text"""
        specs = {}
        
        # Common spec patterns for different product types
        spec_patterns = {
            'processor': ['processor', 'cpu', 'chip', 'core i', 'ryzen', 'snapdragon', 'a17', 'm2'],
            'ram': ['ram', 'memory', 'gb ram', 'gb ddr', 'lpddr'],
            'storage': ['storage', 'ssd', 'nvme', 'gb storage', 'tb storage', 'hard drive'],
            'display': ['display', 'screen', 'inch', 'resolution', 'oled', 'amoled', 'lcd', 'ips'],
            'graphics': ['graphics', 'gpu', 'rtx', 'radeon', 'geforce'],
            'battery': ['battery', 'mah', 'hours', 'battery life'],
            'camera': ['camera', 'mp', 'megapixel', 'lens', 'sensor'],
            'weight': ['weight', 'weighs', 'kg', 'pounds'],
            'connectivity': ['wifi', 'wi-fi', 'bluetooth', '5g', 'thunderbolt', 'usb']
        }
        
        text_lower = text.lower()
        sentences = text.split('.')
        
        for spec_type, keywords in spec_patterns.items():
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    # Extract the relevant part
                    spec_value = sentence.strip()
                    if len(spec_value) > 10 and len(spec_value) < 200:
                        specs[spec_type] = spec_value
                        break
        
        return specs
    
    def _extract_pros_cons(self, text: str) -> tuple:
        """Extract advantages and disadvantages"""
        pros = []
        cons = []
        
        # Positive indicators
        positive_phrases = [
            'excellent', 'great', 'impressive', 'outstanding', 'premium',
            'powerful', 'fast', 'long battery', 'lightweight', 'portable',
            'stunning', 'exceptional', 'high quality', 'reliable', 'affordable'
        ]
        
        # Negative indicators
        negative_phrases = [
            'expensive', 'heavy', 'limited', 'lacks', 'no ', 'poor',
            'slow', 'weak', 'short battery', 'not ideal', 'average'
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) < 20 or len(sentence_lower) > 150:
                continue
            
            # Check for positive aspects
            pos_count = sum(1 for phrase in positive_phrases if phrase in sentence_lower)
            neg_count = sum(1 for phrase in negative_phrases if phrase in sentence_lower)
            
            if pos_count > neg_count and pos_count > 0:
                if len(pros) < 4:
                    pros.append(sentence.strip())
            elif neg_count > pos_count and neg_count > 0:
                if len(cons) < 4:
                    cons.append(sentence.strip())
        
        # If we didn't find enough, add generic ones
        if not pros:
            pros = ["Good performance for intended use", "Reliable build quality"]
        if not cons:
            cons = ["Price may be a consideration", "Specific use case requirements"]
        
        return pros, cons
    
    def _determine_category(self, text: str) -> str:
        """Determine product category"""
        text_lower = text.lower()
        
        categories = {
            'Laptop': ['laptop', 'notebook', 'ultrabook'],
            'Smartphone': ['smartphone', 'phone', 'mobile phone', 'iphone', 'android'],
            'Tablet': ['tablet', 'ipad'],
            'Monitor': ['monitor', 'display screen'], 
            'Camera': ['camera', 'dslr', 'mirrorless'],
            'Headphones': ['headphone', 'earbuds', 'earphone'],
            'Smartwatch': ['smartwatch', 'watch', 'wearable']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'Electronics'
    
    def _extract_product_name(self, text: str) -> str:
        """Extract product name from text"""
        # Usually the product name is in the first sentence
        first_sentence = text.split('.')[0]
        
        # Look for capitalized words that might be product names
        words = first_sentence.split()
        product_name_parts = []
        
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in ['the', 'a', 'an', 'is']:
                product_name_parts.append(word)
                if len(product_name_parts) >= 3:
                    break
        
        if product_name_parts:
            return ' '.join(product_name_parts)
        
        return "Product"
    
    def _determine_use_case(self, text: str, summary: str) -> str:
        """Determine the best use case for the product"""
        text_lower = text.lower() + ' ' + summary.lower()
        
        use_cases = {
            "Professional content creators and designers": ['creator', 'designer', 'professional', 'content creation'],
            "Students and casual users": ['student', 'everyday', 'basic', 'casual'],
            "Gamers and enthusiasts": ['gaming', 'gamer', 'enthusiast', 'high performance'],
            "Business professionals": ['business', 'professional', 'productivity', 'enterprise'],
            "Photography enthusiasts": ['photo', 'photographer', 'photography'],
            "Budget-conscious users": ['budget', 'affordable', 'value for money']
        }
        
        for use_case, keywords in use_cases.items():
            if any(keyword in text_lower for keyword in keywords):
                return use_case
        
        return "General users seeking reliable technology"
    
    def _estimate_price_range(self, text: str) -> str:
        """Estimate price range from description"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['premium', 'flagship', 'pro', 'professional', 'high-end']):
            return "Premium ($1000+)"
        elif any(word in text_lower for word in ['mid-range', 'moderate', 'balanced']):
            return "Mid-range ($500-1000)"
        elif any(word in text_lower for word in ['budget', 'affordable', 'value', 'entry-level']):
            return "Budget ($200-500)"
        else:
            return "Mid-range ($500-1000)"
    


























    
    def compare_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple product summaries and generate comparison insights
        
        Args:
            products: List of structured product summaries
            
        Returns:
            Comparison analysis with common features, differences, and recommendations
        """
        if len(products) < 2:
            return {"error": "At least 2 products required for comparison"}
        
        # Extract all specs
        all_specs = {}
        for product in products:
            if 'key_specs' in product:
                for spec_name in product['key_specs'].keys():
                    if spec_name not in all_specs:
                        all_specs[spec_name] = []
                    all_specs[spec_name].append({
                        'product': product.get('product_name', 'Unknown'),
                        'value': product['key_specs'][spec_name]
                    })
        
        # Find common categories
        categories = [p.get('category', '') for p in products]
        same_category = len(set(categories)) == 1
        
        # Aggregate pros and cons
        all_pros = []
        all_cons = []
        for product in products:
            if 'pros' in product:
                all_pros.extend(product['pros'])
            if 'cons' in product:
                all_cons.extend(product['cons'])
        
        # Generate comparison insights
        comparison = {
            "product_count": len(products),
            "same_category": same_category,
            "category": categories[0] if same_category else "Mixed",
            "products": [
                {
                    "name": p.get('product_name', 'Unknown'),
                    "category": p.get('category', ''),
                    "price_range": p.get('price_range', ''),
                    "best_for": p.get('best_for', '')
                }
                for p in products
            ],
            "spec_comparison": all_specs,
            "summary": self._generate_comparison_summary(products)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, products: List[Dict[str, Any]]) -> str:
        """Generate a model-based comparison summary"""
        if not products:
            return "No products to compare."
        
        # Build comprehensive comparison text for the model
        comparison_text = "Product Comparison:\n\n"
        
        for i, product in enumerate(products, 1):
            comparison_text += f"Product {i}: {product.get('product_name', 'Unknown')}\n"
            comparison_text += f"Category: {product.get('category', 'N/A')}\n"
            comparison_text += f"Price Range: {product.get('price_range', 'N/A')}\n"
            comparison_text += f"Best For: {product.get('best_for', 'N/A')}\n"
            
            # Add key specs
            if 'key_specs' in product and product['key_specs']:
                comparison_text += "Key Specifications: "
                specs = [f"{k}: {v}" for k, v in product['key_specs'].items()]
                comparison_text += ", ".join(specs[:5]) + "\n"
            
            # Add pros
            if 'pros' in product and product['pros']:
                comparison_text += f"Pros: {', '.join(product['pros'][:3])}\n"
            
            # Add cons
            if 'cons' in product and product['cons']:
                comparison_text += f"Cons: {', '.join(product['cons'][:3])}\n"
            
            comparison_text += "\n"
        
        comparison_text += "Compare these products and provide a concise summary highlighting key differences, similarities, and recommendations."
        
        # Use model to generate intelligent summary
        try:
            model_summary = self._safe_summarize(
                comparison_text, 
                max_length=300, 
                min_length=100
            )
            return model_summary
        except Exception as e:
            print(f"Error generating model-based summary: {e}")
            # Fallback to basic summary
            return f"Comparison of {len(products)} products across different specifications and price ranges."

