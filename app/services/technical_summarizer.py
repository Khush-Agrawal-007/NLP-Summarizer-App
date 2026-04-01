# app/services/technical_summarizer.py
"""Technical Product Summarizer.

Uses LangChain for prompt engineering and a local BART model for generation.
Produces structured output: product name, category, key specs, and price range.
"""
import traceback
from typing import Dict, List, Any

from langchain_core.prompts import PromptTemplate

from app.utils.model_utils import load_model_tokenizer, generate_summary


class TechnicalSummarizer:
    """Summarizer for technical products that produces structured output."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        print(f"Loading technical summarizer model: {model_name}")
        self.model, self.tokenizer, self.device = load_model_tokenizer(model_name)
        self.structured_prompt = self._create_structured_prompt()
        print("Technical summarizer initialized.")

    # ------------------------------------------------------------------
    #  Prompt Template
    # ------------------------------------------------------------------

    def _create_structured_prompt(self) -> PromptTemplate:
        """Build the LangChain prompt used for structured product summarization."""
        template = (
            "You are a technical product analyst. Summarize the following product description "
            "into a concise structured format covering: product name, category, key specifications, "
            "and price range.\n\nProduct Description:\n{product_description}\n"
        )
        return PromptTemplate(input_variables=["product_description"], template=template)

    # ------------------------------------------------------------------
    #  Core Summarization
    # ------------------------------------------------------------------

    def _safe_summarize(self, text: str, max_length: int = 400, min_length: int = 200) -> str:
        """Run the model and return the generated summary, or empty string on failure."""
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
            print(f"Summarization error: {e}")
            traceback.print_exc()
            return ""

    def summarize_to_text(self, product_description: str, max_length: int = 200, min_length: int = 50) -> str:
        """Return a plain-text summary of a product description."""
        if not product_description or len(product_description.strip()) < 50:
            return "Input text is too short. Please provide a more detailed product description."

        summary = self._safe_summarize(product_description, max_length=max_length, min_length=min_length)
        return summary or "Unable to generate summary. Please try with a different product description."

    # ------------------------------------------------------------------
    #  Structured Extraction
    # ------------------------------------------------------------------

    def extract_structured_summary(self, product_description: str) -> Dict[str, Any]:
        """Extract a structured summary from a product description.

        Returns a dict with: product_name, category, summary, key_specs, price_range.
        """
        return {
            "product_name": self._extract_product_name(product_description),
            "category":     self._determine_category(product_description),
            "summary":      self.summarize_to_text(product_description, max_length=250),
            "key_specs":    self._extract_specifications(product_description),
            "price_range":  self._estimate_price_range(product_description),
        }

    def _extract_specifications(self, text: str) -> Dict[str, str]:
        """Extract technical specs by matching spec-type keywords to sentences."""
        spec_patterns = {
            'processor':    ['processor', 'cpu', 'chip', 'core i', 'ryzen', 'snapdragon', 'a17', 'm2'],
            'ram':          ['ram', 'memory', 'gb ram', 'gb ddr', 'lpddr'],
            'storage':      ['storage', 'ssd', 'nvme', 'gb storage', 'tb storage', 'hard drive'],
            'display':      ['display', 'screen', 'inch', 'resolution', 'oled', 'amoled', 'lcd', 'ips'],
            'graphics':     ['graphics', 'gpu', 'rtx', 'radeon', 'geforce'],
            'battery':      ['battery', 'mah', 'hours', 'battery life'],
            'camera':       ['camera', 'mp', 'megapixel', 'lens', 'sensor'],
            'weight':       ['weight', 'weighs', 'kg', 'pounds'],
            'connectivity': ['wifi', 'wi-fi', 'bluetooth', '5g', 'thunderbolt', 'usb'],
        }

        specs = {}
        sentences = text.split('.')
        for spec_type, keywords in spec_patterns.items():
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in keywords):
                    value = sentence.strip()
                    # Only store sentences that are a meaningful length
                    if 10 < len(value) < 200:
                        specs[spec_type] = value
                        break  # Use the first matching sentence per spec type

        return specs

    def _determine_category(self, text: str) -> str:
        """Classify the product into a category based on keywords."""
        categories = {
            'Laptop':      ['laptop', 'notebook', 'ultrabook'],
            'Smartphone':  ['smartphone', 'phone', 'mobile phone', 'iphone', 'android'],
            'Tablet':      ['tablet', 'ipad'],
            'Monitor':     ['monitor', 'display screen'],
            'Camera':      ['camera', 'dslr', 'mirrorless'],
            'Headphones':  ['headphone', 'earbuds', 'earphone'],
            'Smartwatch':  ['smartwatch', 'watch', 'wearable'],
        }
        text_lower = text.lower()
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return 'Electronics'

    def _extract_product_name(self, text: str) -> str:
        """Extract the product name from the opening sentence using capitalised words."""
        first_sentence = text.split('.')[0]
        skip_words = {'the', 'a', 'an', 'is'}
        parts = [
            w for w in first_sentence.split()
            if w and w[0].isupper() and w.lower() not in skip_words
        ]
        return ' '.join(parts[:3]) if parts else "Product"

    def _estimate_price_range(self, text: str) -> str:
        """Estimate the price tier from descriptive keywords in the text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['premium', 'flagship', 'pro', 'professional', 'high-end']):
            return "Premium ($1000+)"
        if any(w in text_lower for w in ['mid-range', 'moderate', 'balanced']):
            return "Mid-range ($500-1000)"
        if any(w in text_lower for w in ['budget', 'affordable', 'value', 'entry-level']):
            return "Budget ($200-500)"
        return "Mid-range ($500-1000)"

    # ------------------------------------------------------------------
    #  Product Comparison
    # ------------------------------------------------------------------

    def compare_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple structured product summaries.

        Returns a unified spec table and a model-generated comparison narrative.
        """
        if len(products) < 2:
            return {"error": "At least 2 products required for comparison"}

        # Merge specs from all products into a single comparison table
        spec_comparison: Dict[str, list] = {}
        for product in products:
            for spec_name, spec_value in product.get('key_specs', {}).items():
                spec_comparison.setdefault(spec_name, []).append({
                    'product': product.get('product_name', 'Unknown'),
                    'value': spec_value,
                })

        categories = [p.get('category', '') for p in products]
        same_category = len(set(categories)) == 1

        return {
            "product_count":   len(products),
            "same_category":   same_category,
            "category":        categories[0] if same_category else "Mixed",
            "products": [
                {
                    "name":        p.get('product_name', 'Unknown'),
                    "category":    p.get('category', ''),
                    "price_range": p.get('price_range', ''),
                }
                for p in products
            ],
            "spec_comparison": spec_comparison,
            "summary":         self._generate_comparison_summary(products),
        }

    def _generate_comparison_summary(self, products: List[Dict[str, Any]]) -> str:
        """Build a model-generated comparison narrative from the structured product data."""
        if not products:
            return "No products to compare."

        lines = ["Product Comparison:\n"]
        for i, product in enumerate(products, 1):
            lines.append(f"Product {i}: {product.get('product_name', 'Unknown')}")
            lines.append(f"  Category:    {product.get('category', 'N/A')}")
            lines.append(f"  Price Range: {product.get('price_range', 'N/A')}")
            specs = product.get('key_specs', {})
            if specs:
                spec_str = ', '.join(f"{k}: {v}" for k, v in list(specs.items())[:5])
                lines.append(f"  Specs: {spec_str}")
            lines.append("")

        lines.append("Provide a concise comparison highlighting key differences and similarities.")
        comparison_text = "\n".join(lines)

        result = self._safe_summarize(comparison_text, max_length=300, min_length=100)
        return result or f"Comparison of {len(products)} products across different specifications and price ranges."
